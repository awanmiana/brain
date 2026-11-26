#!/usr/bin/env python3
"""Brain11 chunked trainer — stop any time, resume exactly where you left off.

Every checkpoint is written atomically (tmp file + fsync + os.replace):
    brain11_<name>_weights.pt   model weights (best + latest)
    brain11_<name>_optim.pt     optimizer state
    brain11_<name>_state.json   step, tokens seen, data hashes, config
    brain11_<name>_log.jsonl    append-only training log
    brain11_<name>_best.pt      best validation checkpoint

The corpus is read-only and never deleted. If a run is interrupted, the last
completed chunk (default 100 steps, saved every 50) is intact; rerun with the
same command to resume.

Usage:
    python3 brain11_train.py --model all --steps 1200 --chunk-steps 100
    python3 brain11_train.py --model coder --steps 1200 --stop-after-chunk
    python3 brain11_train.py --model language --steps 600 --fresh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from brain11_model import Brain, Brain11Config
from brain11_tokenizer import TOKENIZER_PATH, encode

HERE = Path(__file__).resolve().parent

CORPORA = {
    "coder": HERE / "brain11_code.jsonl",
    "language": HERE / "brain11_language.jsonl",
    "agentic": HERE / "brain11_agentic.jsonl",
}


# ------------------------------------------------------------------ data

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_docs(path: Path) -> list[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
    return docs


def build_arrays(name: str, val_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray, dict]:
    """Encode documents into uint16 arrays, cached by content hash."""
    corpus = CORPORA[name]
    if not corpus.exists():
        raise FileNotFoundError(f"missing corpus: {corpus} (run brain11_prepare.py)")
    corpus_hash = sha256_file(corpus)
    tok_hash = sha256_file(TOKENIZER_PATH)
    selflearn = HERE / "brain11_selflearn.jsonl"
    selflearn_hash = sha256_file(selflearn) if (name == "coder" and selflearn.exists()) else None
    meta_path = HERE / f"brain11_{name}_data.json"
    train_path = HERE / f"brain11_{name}_train.npy"
    val_path = HERE / f"brain11_{name}_val.npy"
    meta = {
        "corpus": corpus.name, "corpus_sha256": corpus_hash,
        "tokenizer_sha256": tok_hash, "val_frac": val_frac,
        "selflearn_sha256": selflearn_hash, "version": 4,
    }

    if meta_path.exists() and train_path.exists() and val_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in meta.items()):
            train = np.load(train_path)
            val = np.load(val_path)
            print(f"[data] {name}: cached train={len(train):,} val={len(val):,} tokens")
            return train, val, old

    docs = load_docs(corpus)
    if selflearn_hash:
        extra = load_docs(selflearn)
        cap = max(1, int(len(docs) * 0.25))
        docs.extend(extra[:cap])
        print(f"[data] {name}: +{min(len(extra), cap)} verified self-learned examples")
    if not docs:
        raise ValueError(f"empty corpus: {corpus}")
    order = list(range(len(docs)))
    random.Random(1234).shuffle(order)
    n_val = max(1, int(len(docs) * val_frac))
    val_idx = set(order[:n_val])

    train_ids, val_ids = [], []
    t0 = time.time()
    for i, doc in enumerate(docs):
        ids = encode(doc)
        if i in val_idx:
            val_ids.extend(ids)
        else:
            train_ids.extend(ids)
    train = np.asarray(train_ids, dtype=np.uint16)
    val = np.asarray(val_ids, dtype=np.uint16)
    # np.save appends ".npy" when the name does not end in it, which breaks an
    # os.replace on "<name>.npy.tmp". Write through a file handle instead.
    for arr, path in ((train, train_path), (val, val_path)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    meta.update({"docs": len(docs), "train_tokens": int(len(train)),
                 "val_tokens": int(len(val)), "encoded_seconds": round(time.time() - t0, 2)})
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    print(f"[data] {name}: {len(docs):,} docs -> train={len(train):,} val={len(val):,} tokens "
          f"({time.time() - t0:.1f}s)")
    return train, val, meta


# ------------------------------------------------------------------ checkpoint helpers

def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path):
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# ------------------------------------------------------------------ training

def lr_at(step: int, total: int, base_lr: float, warmup: int, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def evaluate(model: Brain, val: np.ndarray, seq_len: int, device, max_windows: int = 24) -> float:
    model.eval()
    rng = random.Random(999)
    nll, count = 0.0, 0
    with torch.no_grad():
        for _ in range(min(max_windows, max(1, len(val) // seq_len))):
            p = rng.randrange(0, max(1, len(val) - seq_len - 1))
            x = torch.from_numpy(val[p:p + seq_len].astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy(val[p + 1:p + seq_len + 1].astype(np.int64)).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits, loss = model(x, y)
            nll += float(loss.item()) * seq_len
            count += seq_len
    model.train()
    return math.exp(nll / max(1, count))


def train_one(name: str, args, device: str):
    corpus = CORPORA[name]
    if not corpus.exists():
        print(f"[{name}] missing {corpus.name}; run brain11_prepare.py first")
        return False
    free_gb = shutil.disk_usage(HERE).free / (1024 ** 3)
    if free_gb < 2.0:
        print(f"[{name}] only {free_gb:.1f} GB free disk; refusing to start (need 2 GB)")
        return False
    if device == "cuda":
        # leave headroom for the desktop/other apps on the 8 GB card
        torch.cuda.set_per_process_memory_fraction(0.90)

    train_ids, val_ids, data_meta = build_arrays(name, args.val_frac)
    if len(train_ids) < args.seq_len + 2:
        print(f"[{name}] corpus too small")
        return False

    cfg = Brain11Config()
    model = Brain(cfg).to(device)
    n_params = model.n_params()
    print(f"[{name}] {n_params:,} params, train={len(train_ids):,} tokens, device={device}")

    decay, no_decay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    weights_path = HERE / f"brain11_{name}_weights.pt"
    optim_path = HERE / f"brain11_{name}_optim.pt"
    state_path = HERE / f"brain11_{name}_state.json"
    log_path = HERE / f"brain11_{name}_log.jsonl"
    best_path = HERE / f"brain11_{name}_best.pt"

    step = 0
    tokens_seen = 0
    best_val = float("inf")
    if not args.fresh and state_path.exists() and weights_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("data_sha256") not in (None, data_meta["corpus_sha256"]):
            print(f"[{name}] data changed since checkpoint; use --fresh to start over")
            return False
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if optim_path.exists():
            opt.load_state_dict(torch.load(optim_path, map_location=device, weights_only=False))
        step = int(state.get("step", 0))
        tokens_seen = int(state.get("tokens_seen", 0))
        best_val = float(state.get("best_val", float("inf")))
        print(f"[{name}] resumed at step {step}/{args.steps} ({tokens_seen:,} tokens seen)")
        if step >= args.steps:
            print(f"[{name}] already at target steps; nothing to do (pass larger --steps to continue)")
            return True

    use_amp = device == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    seq_len = min(args.seq_len, cfg.ctx)
    model.train()
    chunk_start_step = step
    chunk_start_time = time.time()
    running_loss = 0.0
    running_count = 0
    last_save = step

    def save(step_now: int, val_ppl: float | None = None, final: bool = False):
        nonlocal last_save, best_val
        atomic_torch_save({
            "config": cfg.__dict__,
            "step": step_now,
            "state_dict": model.state_dict(),
            "n_params": n_params,
        }, weights_path)
        atomic_torch_save(opt.state_dict(), optim_path)
        if val_ppl is not None and val_ppl < best_val:
            best_val = val_ppl
            atomic_copy(weights_path, best_path)
        state = {
            "model": name,
            "step": step_now,
            "tokens_seen": tokens_seen,
            "best_val": best_val,
            "last_val_ppl": val_ppl,
            "config": cfg.__dict__,
            "n_params": n_params,
            "corpus": corpus.name,
            "data_sha256": data_meta["corpus_sha256"],
            "tokenizer_sha256": data_meta["tokenizer_sha256"],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "final": final,
        }
        atomic_json_save(state, state_path)
        last_save = step_now

    try:
        while step < args.steps:
            lr = lr_at(step, args.steps, args.lr, args.warmup, args.min_lr)
            for group in opt.param_groups:
                group["lr"] = lr

            rng = random.Random((args.seed * 1000003) + step * 7919)
            xs, ys = [], []
            for _ in range(args.batch):
                p = rng.randrange(0, len(train_ids) - seq_len - 1)
                xs.append(train_ids[p:p + seq_len].astype(np.int64))
                ys.append(train_ids[p + 1:p + seq_len + 1].astype(np.int64))
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            step += 1
            tokens_seen += x.numel()
            running_loss += float(loss.item())
            running_count += 1

            if step % args.log_every == 0 or step == args.steps:
                avg = running_loss / max(1, running_count)
                elapsed = max(1e-6, time.time() - chunk_start_time)
                tok_s = (tokens_seen - int(data_meta.get("tokens_seen_at_resume", 0))) / elapsed
                print(f"[{name}] step {step}/{args.steps} loss={avg:.4f} lr={lr:.2e} "
                      f"{tok_s:,.0f} tok/s", flush=True)

            if step % args.save_every == 0 or step % args.chunk_steps == 0 or step == args.steps:
                val_ppl = None
                if step % max(args.save_every, args.eval_every) == 0 or step == args.steps:
                    val_ppl = evaluate(model, val_ids, seq_len, device)
                    print(f"[{name}] eval step {step} val PPL={val_ppl:.2f}", flush=True)
                save(step, val_ppl, final=(step == args.steps))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": step, "loss": round(running_loss / max(1, running_count), 5),
                        "val_ppl": None if val_ppl is None else round(val_ppl, 4),
                        "lr": lr, "tokens_seen": tokens_seen,
                        "elapsed": round(time.time() - chunk_start_time, 2),
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                running_loss = 0.0
                running_count = 0

            if step - chunk_start_step >= args.chunk_steps:
                if args.stop_after_chunk:
                    print(f"[{name}] chunk complete at step {step} "
                          f"({time.time() - chunk_start_time:.1f}s); stopping as requested")
                    return True
                chunk_start_step = step
                chunk_start_time = time.time()
                data_meta["tokens_seen_at_resume"] = tokens_seen

    except KeyboardInterrupt:
        print(f"\n[{name}] interrupted at step {step}; saving atomically...", flush=True)
        save(step)
        print(f"[{name}] saved. rerun to resume.", flush=True)
        return True
    except torch.cuda.OutOfMemoryError:
        print(f"\n[{name}] CUDA OOM at step {step}; saving checkpoint.", flush=True)
        save(step)
        torch.cuda.empty_cache()
        print(f"[{name}] rerun with --batch 4 --seq-len 256 to continue safely.", flush=True)
        return False
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"\n[{name}] OOM at step {step}; saving checkpoint.", flush=True)
            save(step)
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"[{name}] rerun with --batch 4 --seq-len 256 to continue safely.", flush=True)
            return False
        raise

    val_ppl = evaluate(model, val_ids, seq_len, device)
    save(step, val_ppl, final=True)
    print(f"[{name}] done at step {step}, val PPL={val_ppl:.2f}, weights -> {weights_path.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Brain11 chunked trainer")
    ap.add_argument("--model", choices=["coder", "language", "agentic", "all"], default="all")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=320)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--chunk-steps", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--stop-after-chunk", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        import sys
        print(f"[error] --device cuda requested but this interpreter's torch has no CUDA.\n"
              f"        interpreter: {sys.executable}\n"
              f"        torch: {torch.__version__} (cuda {torch.version.cuda})\n"
              f"        run with the Python that has the CUDA build of torch.")
        return 1
    names = ["coder", "language", "agentic"] if args.model == "all" else [args.model]
    for name in names:
        ok = train_one(name, args, device)
        if not ok:
            return 1
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
