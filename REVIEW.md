# Review, 2026-11-11: what this brain is, and what to keep

Written after reading every file, running the repository's own checks
(`pathway_lattice.py --verify`: all pass) and measuring the points below.
`review_probe.py <this folder>` reproduces the lattice measurements. The
menu evaluation used a private application's menu; its request set is kept
with that application, not here.

## What the Pathway and Triple Lattice are

A store of `(subject, relation) -> object` facts. Each token gets an
unrelated random code from a hash; a fact is one read vector and one write
vector; a query is answered by the nearest read vector, or `UNKNOWN` under a
threshold. Measured on the 138 compiled keys:

| | lattice | plain dictionary |
|---|---|---|
| answers | 138 of 138 the same | - |
| time per query | tens of milliseconds | under a microsecond |
| size on disk | 2.2 MB | 4.8 KB |

Because the codes are unrelated, only the exact tokens it was compiled with
ever route: `France`, `frnace`, `capitals` and `capital-city` all return
`UNKNOWN`, and the codes of `camera` and `cameras` have cosine 0.03. The
noise it is robust to is noise added to a vector, which a real query never
has. "2,000 parameters per fact" is storage, not capacity: nothing is
generalised, so it is not comparable to the parameters of a language model.

## What is worth keeping

1. **Determine, do not generate.** One answer when it is clear, a short
   list when it is not, `UNKNOWN` otherwise, and never a guess. This is the
   same rule a careful agent system already lives by.
2. **Grow by compiling.** A new fact or phrasing is added in one step and is
   usable at once, and can be removed again.
3. **Exact integer arithmetic**, so the same input gives the same answer on
   any machine.

## The change that makes it useful: keys that mean something

`meaning_lattice.py` keeps the loop (encode, route by exact integer cosine,
abstain, determine, grow) and replaces the encoder: text becomes weighted
stems, word pairs and letter triples, with a small synonym table and a
misspelling mender. Tried on a real application menu (261 places), with requests
written the way a person types them, never run before the model was fixed:

| fresh set, 81 requests in scope | exact words only | meaning keys |
|---|---|---|
| right place first | 70% | 79% |
| right place in the first three | 81% | 88% |

With the gates set on a separate half: 41 asserted, 40 of them right; 29
offered as a short list of three, the right place in 27; 11 `UNKNOWN`. Of 15
requests the menu has no place for, 12 abstain, 2 get a short list, 1 is
asserted wrongly. 3.5 ms per request, standard library only.

What it does not do: meaning that is not in the words. A request phrased
by its effect ("turn the alerts off") does not find a place named by its
mechanism ("Disarm notifications"). Teaching it one
person's phrasings did not help with other places (78% before and after).
That boundary is where a language model is needed, as a proposer only.

## Recommendation for an agent system built on this

- Orchestration is playbooks: trigger, condition, step, check, next step.
  That needs no learned parameters and should stay deterministic.
- Use the meaning lattice to find a place from what a person typed, as a
  suggestion with a short list. It never runs anything by itself.
- Keep facts in an exact store (SQLite), with where each came from.
- A language model, when one is configured, proposes; the lattice and the
  permission checks decide.
- The affective engine and the child brain are not needed for this.
