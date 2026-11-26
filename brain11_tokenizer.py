#!/usr/bin/env python3
"""Brain11 tokenizer — byte-level BPE with a 4096-token vocab.

Trains with the fast `tokenizers` library, but the saved tokenizer.json is a
normal HuggingFace tokenizer file. Runtime encoding/decoding works through the
same file (Rust-backed, no Python BPE loop).

Special ids (stable):
    0 = <pad>, 1 = <bos>, 2 = <eos>, 3 = <unk>
Byte tokens and merges occupy 4..4095.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
TOKENIZER_PATH = HERE / "brain11_tokenizer.json"
VOCAB_SIZE = 4096
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

_TOKENIZER = None


def train_tokenizer(texts: Iterable[str], out_path: Path = TOKENIZER_PATH,
                    vocab_size: int = VOCAB_SIZE) -> "object":
    """Train a byte-level BPE tokenizer and save it as JSON."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return tok


def get_tokenizer(path: Path = TOKENIZER_PATH):
    """Load and cache the tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        if not path.exists():
            raise FileNotFoundError(f"tokenizer not found: {path}")
        _TOKENIZER = Tokenizer.from_file(str(path))
    return _TOKENIZER


def encode(text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """Encode text to ids (bytes-level BPE)."""
    ids = get_tokenizer().encode(text, add_special_tokens=False).ids
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids


def decode(ids: list[int], skip_special: bool = True) -> str:
    """Decode ids back to text. Special tokens are dropped by default."""
    return get_tokenizer().decode(list(ids), skip_special_tokens=skip_special)


def info() -> dict:
    """Small summary for docs/logs."""
    tok = get_tokenizer()
    return {
        "path": str(TOKENIZER_PATH),
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": SPECIALS,
    }


if __name__ == "__main__":
    print(json.dumps(info(), indent=2))
