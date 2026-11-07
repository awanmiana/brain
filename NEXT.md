# NEXT — handoff state for continuing this work

> Read this first in a new session. It records what exists, what is verified,
> what to build next, and the environment traps. Full defect/gap detail is in
> [`AUDIT.md`](AUDIT.md).

## Status snapshot

- Branch `main`, latest commit at handoff: `0ff7d18` (Pathway Lattice v1).
- **23/23 automated checks pass**: `python3 pathway_lattice.py --verify` (~46 s).
- The model file `pathway_lattice_model.json` is **git-ignored**; every entry
  point auto-trains (~40 s) if it is missing, or run `--train` explicitly.
- No numpy / pytest / network. **Pure Python 3.14 stdlib only.** Keep it so.

## Environment traps (cost hours if forgotten)

- In this proot/Termux container, **shell pipes hang** on longer-running
  Python commands (`... | tail -n` never returns). Redirect to a file instead:
  `python3 x.py --verify > /tmp/opencode/v.log 2>&1; tail -20 /tmp/opencode/v.log`
- `timeout` does not reliably kill processes inside proot. Prefer the
  file-redirect pattern; avoid long pipelines.
- The box occasionally reboots under load; re-check `uptime` if things stall.
- Never introduce `numpy`, `torch`, RNG state, dict/set iteration order
  dependence, or floating-point into the lattice path — determinism and
  bit-exactness are the product. Floats exist **only** in the reliability
  probability column (documented in AUDIT.md item 5).

## Verified facts (do not re-derive, but re-run `--verify` after changes)

| claim | value |
|---|---|
| compiled facts/tools | 139 triples + 100 arithmetic = 239 operators |
| learned params | 239 × 2,000 = 478,000 (ceiling 500 ops = 1,000,000) |
| routing threshold | 0.8098 (margin: positives 1.000, hard negatives 0.8093) |
| state hash | `a55e1006b2aa5f07…` (full hash printed by `--verify`/`--report`) |
| direct QA / arithmetic | exact, 100/100 |
| 3-hop / 4-hop chains | `alexander` / `erin`, correct terminals |
| OOD assertions | 0 / 821 (raw argmax answers all 821, all wrong) |
| asserted-answer precision | 1.000 at every noise level |
| calibration ECE | 0.264 raw cosine → 0.000 calibrated |
| S0 tier | 1,000 params, denoising MSE 0.752 → 0.320 (2.35×) |
| growth | +2,000 params/fact; 2,000-fact stream: 478K → 4.478M |

## What exists (file map)

| file | role |
|---|---|
| `pathway_lattice.py` | core engine: fixed-point kernel, `PathOp` (rank-1, 2,000 params), routing/`walk`, `RankPath` (rank-L fit), calibration, `learn_fact`, CLI (`--train/--verify/--demo/--bench/--query/--report`) |
| `triple_lattice.py` | 3-stage determination (domain/relation/value over 10⁹ cells), reliability table, verdicts, `--experiment` precision tables |
| `hybrid_demo.py` | zero-parameter English grammar front-end → lattice queries |
| `grow.py` | self-growth loop demo (`--demo`, `--stream N`) |
| `pathway_seed.py` | seed corpus (`FACTS`, `TOOL_FACTS`, `BINARY_RELATIONS`) |
| `PATHWAY_LATTICE.md`, `TRIPLE_LATTICE.md` | architecture and measured results |
| `AUDIT.md` | 10 fixed defects + 12 open gaps, prioritized |

## Next work — in priority order

### 1. HIGH — per-stage calibration (AUDIT.md gap 1)
`triple_lattice.py`: `Reliability` is fitted on operator-level scores in
`calibration_samples()`, but `TripleLattice._stage()` applies `self.rel` to
*domain-max* and *relation-max* scores. Fix: fit three tables
(`dom_rel`, `rel_rel`, `val_rel`) from per-stage score samples (extend
`calibration_samples` to record stage scores), or calibrate each stage on its
own score distribution. Re-run `--calibrate` + `--experiment`; ECE and
asserted-precision must not regress.

### 2. HIGH — relation typing instead of blanket transitive closure (gap 2)
`pathway_lattice.py` `Lattice.walk()` chains any relation. Add relation
metadata (e.g. in `pathway_seed.py`: `TRANSITIVE = {"teaches","knows","is-a"}`,
`FUNCTIONAL = {"capital","symbol","orbits",...}`) and enforce a max-hop /
transitivity policy in `determine_path` (triple_lattice `determine()`).
Known symptom: `moon orbits → earth → sun`. Expected after fix: direct answer
`earth`; chain only for relations marked transitive. Add verify checks.

### 3. MEDIUM — routing index (gap 3)
`Lattice.route`/`best_match` are O(P) linear scans (~9 ms at 239 ops). For
10⁵+ facts add an LSH / random-projection index over `op.v` with exact
re-scoring of candidates. Keep exact argmax determinism (index returns a
superset, exact scores decide).

### 4. MEDIUM — persist calibration with the model (gap 4)
`triple_lattice.py` refits reliability on every startup (~17 s). Save the
bin counts/hits in the model JSON (or a sidecar) and load them; expose
`--recalibrate`.

### 5. MEDIUM — retraction API (gap 6)
Add `retract_fact(lat, s, r, o)` (remove operator, optionally recalibrate) and
a policy for contradiction replacement. Grow-only is not enough for agents.

### 6. MEDIUM — binary persistence (gap 8)
JSON is ~7 KB/fact (~7 GB at 1M facts). Needs packed binary (e.g.
`array('q')` + header) before scale.

### 7. Lower priority
Candidate indexing for `decode` at 10⁴+ symbols; OOD score separate from the
cosine gate; explicit `allow_multiple` config rather than corpus-derived;
replace `hybrid_demo` grammar with a real (small or borrowed) parser.

## Invariants to preserve

- Same input → same operator path → bit-identical `Determination.to_json()`.
- `--verify` must stay green; add a check for every new behaviour.
- Parameter accounting stays exact: 2,000 learned params per operator
  (read 1000 + write 1000); S1 ceiling 500 ops = 1,000,000.
- Abstention is a feature: never return a confident wrong answer; extend the
  negative suite when adding relations.

## Suggested first commands

```bash
cd /root/brain && uptime
python3 pathway_lattice.py --verify > /tmp/opencode/v.log 2>&1; tail -30 /tmp/opencode/v.log
python3 pathway_lattice.py --train
python3 triple_lattice.py --experiment > /tmp/opencode/e.log 2>&1; tail -25 /tmp/opencode/e.log
```
