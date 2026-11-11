"""Probes of the brain repository: what the lattice is equivalent to, and
where it stops.  Usage: probe.py <repo dir>"""
import json, sys, time
repo = sys.argv[1]
sys.path.insert(0, repo)
import os
os.chdir(repo)
import pathway_lattice as pl
from pathway_seed import FACTS, TOOL_FACTS

t0 = time.time()
if not os.path.exists(pl.MODEL_FILE):
    # the module trains on demand through its CLI helpers
    os.system(f"{sys.executable} pathway_lattice.py --train > /dev/null 2>&1")
lat = pl.Lattice.load(pl.MODEL_FILE)
print("ops", len(lat.ops), "params", lat.param_count(), "load/train s", round(time.time()-t0, 1))
print("model file bytes", os.path.getsize(pl.MODEL_FILE))

cb = lat.codebook
def ask(s, r, hops=False):
    h = pl.normalize([a + b for a, b in zip(cb.vec(s), cb.vec(pl.REL + r))])
    d = lat.determine_path(h) if hops else lat.determine_next(h)
    return d.value, round(d.cos / pl.ONE, 3), d.path

# 1. The same answers from a plain dict.
table = {}
for s, r, o in FACTS + TOOL_FACTS:
    table.setdefault((s, r), []).append(o)
same = wrong = 0
t1 = time.time()
for (s, r), objs in table.items():
    v, c, p = ask(s, r)
    if v in objs: same += 1
    else: wrong += 1
lat_ms = (time.time() - t1) * 1000 / len(table)
t2 = time.time()
for _ in range(1000):
    for key in table: table.get(key)
dict_us = (time.time() - t2) * 1e6 / (1000 * len(table))
print(f"dict agreement: {same}/{same+wrong}; lattice {lat_ms:.2f} ms/query, dict {dict_us:.3f} us/query")
print("dict json bytes", len(json.dumps([list(k) + v for k, v in table.items()])))

# 2. Does it generalise to wording it has not seen?  (case, typo, plural, synonym)
s0, r0, o0 = "france", "capital", "paris"
for s, r in [("france", "capital"), ("France", "capital"), ("frnace", "capital"),
             ("france", "capitals"), ("france", "capital-city"), ("french-republic", "capital"),
             ("germany", "capital"), ("germany", "symbol")]:
    print(f"  {s!r:20} {r!r:16} -> {ask(s, r)}")

# 3. How similar are the codes of similar words?  (0 = unrelated)
for a, b in [("paris", "Paris"), ("capital", "capitals"), ("camera", "cameras"), ("offline", "down")]:
    print(f"  cos({a},{b}) = {pl.cos_fx(cb.vec(a), cb.vec(b)) / pl.ONE:+.3f}")
