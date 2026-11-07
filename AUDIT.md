# AUDIT — Pathway / Triple Lattice v1

Critical self-audit performed after the first release of the code. The purpose
is to answer honestly: *is this top-notch and complete?* No. Below is what was
found, what was fixed, what remains open, and what is actually verified.

Status: **23/23 automated checks pass** (`python3 pathway_lattice.py --verify`).
The model file `pathway_lattice_model.json` is a reproducible build artifact
(`--train`, ~40 s) and is git-ignored.

---

## 1. Defects found and fixed in this audit

| # | severity | defect | effect | fix |
|---|---|---|---|---|
| 1 | **critical** | Noise generator used amplitude `level/8` of ONE while normalized coordinates are ~`ONE/32` | "mild" noise was 16× the signal; every measurement at noise ≥ 0.04 collapsed to abstention | amplitude now `level/512` (matches coordinate scale); verified curve: cos 1.00 → 0.70 → 0.54 |
| 2 | **critical** | `ChannelGainModel` never set `self.dim` | tier-S0 benchmark crashed with `AttributeError`; the S0 claim in the docs was untested | `self.dim = dim`; S0 is now in `--verify` |
| 3 | **critical** | S0 `apply()` re-normalised the output while targets are un-normalised | learned gains made MSE *worse* (0.752 → 0.949) | removed re-normalisation; measured **0.752 → 0.320 (2.35×)** |
| 4 | high | `learn_fact` treated genuinely multi-valued relations (`has-moon`) as contradictions | legitimate second answers (`deimos`) were rejected | relation cardinality auto-derived from the corpus (`MULTI_VALUED`); status `learned-multi` |
| 5 | high | `calibrate_route_threshold` silently fell back when positives and negatives overlapped | a bad threshold could ship with no signal | returns `separable`; `--train` prints a warning; fallback documented |
| 6 | medium | routing recomputed `‖v‖` for the winner on every call | `--verify` took ~3× longer than necessary | `PathOp.vn` cached at construction; verify 46 s |
| 7 | medium | `hybrid_demo` parsed "what does the weather tool need" as the intent, not the tool's needs | wrong query | "need"-questions map intent → tool → needs |
| 8 | low | `grow.py` used `input()` for the save prompt | blocked non-interactive runs | `--save` flag |
| 9 | low | `_stage` crashed on an empty value cell | potential `IndexError` | guarded with an explicit `UNKNOWN` placeholder |
| 10 | low | Bench noise table was hard-coded to a stale formula | misleading robustness numbers | unified with the corrected generator |

## 2. Open gaps (known, not yet fixed)

### HIGH

1. **Stage-1/2 calibration mismatch.** The reliability table is fitted on
   *operator-level* scores, then applied to *domain-max* and *relation-max*
   scores, whose distributions are stochastically higher. The verdict
   probabilities for the first two stages are therefore optimistic.
   *Fix:* fit one reliability table per stage, or fit on per-stage scores.
2. **Transitive closure is applied to every relation.** A walk from
   `moon orbits` continues to `sun` because the output state is still a valid
   match for another `orbits` operator. Correct for `teaches`/`knows`/`is-a`;
   semantically wrong for non-transitive relations.
   *Fix:* relation typing (transitive / functional / multi-valued) and a
   `max_hops` policy per relation — `ESTIMATE` already labels the result, but
   the label is doing work the model should do.

### MEDIUM

3. **Routing is O(P) linear scan.** Fine at 500 ops (~9 ms/query), but
   scaling to 10⁵–10⁶ facts needs an ANN/LSH index over the read vectors.
   The "2B params = 1M facts" story is storage-true but *routing* does not yet
   scale that far.
4. **Calibration is refit at every startup** (≈17 s) and not persisted with
   the model. It is deterministic, but wasteful and easy to diverge from the
   saved artifact.
5. **Float determinism scope.** The lattice is pure integer and bit-exact.
   The reliability table and verdict probabilities use IEEE-754 floats —
   deterministic on conforming platforms, but the "bit-identical everywhere"
   claim is about the lattice, not the float-derived probability column.
6. **No retraction API.** `learn_fact` grows only. Conflicts are flagged and
   the first-compiled fact wins, but there is no `retract_fact`/update policy.
   An agent that must forget (GDPR-style) needs one.
7. **Calibration negatives are corpus-derived.** Threshold quality is only as
   good as the generated negative set; there is no dedicated OOD detector
   beyond the cosine gate.
8. **JSON persistence cost.** ~7 KB/fact as text JSON; 1M facts ≈ 7 GB.
   Needs a packed binary format before real-scale deployment.

### LOW

9. `decode` is O(candidates) per query; fine for hundreds of symbols, needs an
   index for 10⁴+.
10. `hybrid_demo` is a deterministic toy grammar, not an NLU front-end.
11. Multi-valued auto-detection is corpus-derived; a relation never seen with
    two objects defaults to functional.
12. The parameter-count framing ("2,000 per fact") counts read+write vectors;
    gains/bias are folded in. Correct as stated, but worth stating explicitly.

## 3. Claims verified TRUE by `--verify`

- Fixed-point kernel: cosines bounded, normalisation within 1 LSB.
- Every compiled triple routes (139/139); **0/4,101** hard negatives route.
- Direct QA exact; 3-hop and 4-hop chain terminals correct.
- Arithmetic table 100/100 exact.
- Unknown triples abstain; wrong-relation arithmetic abstains.
- Independent rebuild → identical state hash (bit-exact determinism).
- Repeated inference byte-identical.
- Parameter accounting: S0 = 1,000; S1 ceiling = 1,000,000; 239 × 2,000.
- Rank-4 path fits 4 pairs exactly (MSE 1.2e-06); rank-2 underfits as designed.
- S0 gains reduce denoising MSE 2.35× (real measurement, replacing a previously
  unverified doc claim).
- Growth loop: +2,000 params/fact, conflict and multi-valued policies behave.

## 4. Verdict

The core (deterministic routing, composition, abstention, calibration,
growth loop) is **sound and verified within its claims**. The design is *not*
complete: items 1–2 are correctness-of-semantics issues that should be fixed
before claiming general multi-hop reasoning, and items 3–8 are the real work
between "impressive prototype" and "deployable system".
