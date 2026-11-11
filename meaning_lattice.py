"""Meaning Lattice (prototype) - the Pathway Lattice with keys that mean
something.

The Pathway Lattice routes a query to the operator whose read vector is
nearest, abstains under a threshold, and writes that operator's answer.
Its codebook gives every token an unrelated random code, so only the exact
tokens it was compiled with ever route.  Here the same loop is kept -
encode, route by exact integer cosine, abstain, determine, grow by
compiling - and the encoder is replaced: text becomes a sparse vector of
weighted features (stems, word pairs, letter triples, a small synonym table)
so that wording the model has never seen still lands near the right key.

Pure standard library, integer arithmetic only: the same input gives the
same answer on any machine.
"""
from __future__ import annotations

import hashlib
import json
import re
from math import isqrt

ONE = 1 << 12
WORD = re.compile(r"[a-z0-9]+")
#: Words that carry almost no meaning in a request.
FILLER = frozenset("""a an the of to for on in at is are be me my i we you it
this that these those please can could would will do does did show give get
see tell let want need have has just now all some any with and or by from
about there here""".split())


def stem(word: str) -> str:
    for tail, put in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("ed", ""),
                      ("es", ""), ("s", "")):
        if word.endswith(tail) and len(word) - len(tail) >= 3:
            return word[: len(word) - len(tail)] + put
    return word


class MeaningLattice:
    def __init__(self, synonyms: dict | None = None) -> None:
        self.synonyms = dict(synonyms or {})   # stem -> canonical stem
        self.keys: list = []                   # (target, vector, norm2, text)
        self.df: dict = {}
        self.docs = 0
        self._vocabulary: list = []
        self._gram_cache: dict = {}
        self._middle = ONE
        self._texts: list = []                 # (target, text) as compiled

    # -- encoding -------------------------------------------------------
    def words(self, text: str, *, mend: bool = False) -> list:
        found = []
        for raw in WORD.findall(str(text).lower()):
            word = stem(raw)
            word = self.synonyms.get(word, self.synonyms.get(raw, word))
            if mend and word not in self.df and word not in FILLER:
                word = self._mend(word)
            found.append(word)
        return found

    @staticmethod
    def _grams(word: str) -> frozenset:
        padded = f"^{word}$"
        return frozenset(padded[i:i + 3] for i in range(len(padded) - 2))

    def _mend(self, word: str) -> str:
        """A word no key has: the known word it is most plainly a
        misspelling of (letter triples shared, Dice >= 0.6), else itself."""
        if len(word) < 4:
            return word
        mine = self._grams(word)
        best, best_score = word, 0
        for known in self._vocabulary:
            if abs(len(known) - len(word)) > 2:
                continue
            theirs = self._gram_cache.get(known)
            if theirs is None:
                theirs = self._gram_cache[known] = self._grams(known)
            score = (2000 * len(mine & theirs)) // (len(mine) + len(theirs))
            if score > best_score or (score == best_score and known < best):
                best, best_score = known, score
        return best if best_score >= 600 else word

    def _weight(self, word: str) -> int:
        """sqrt(N / df), as an exact integer; a word no key has seen weighs
        like a common one (it cannot match, it can only dilute)."""
        df = self.df.get(word, 0)
        if word in FILLER:
            return ONE // 8
        if df == 0:
            # A word the catalog has never seen is evidence the request is
            # about something else: it weighs like a middling known word.
            return self._middle
        return isqrt((max(self.docs, 1) << 24) // df)

    def encode(self, text: str, *, mend: bool = False) -> dict:
        words = self.words(text, mend=mend)
        vector: dict = {}
        for word in words:
            weight = self._weight(word)
            vector["w:" + word] = vector.get("w:" + word, 0) + weight
            padded = f"^{word}$"
            grams = [padded[i:i + 3] for i in range(len(padded) - 2)]
            if word not in FILLER:
                share = weight // (2 * max(len(grams), 1))
                for gram in grams:
                    vector["c:" + gram] = vector.get("c:" + gram, 0) + share
        solid = [word for word in words if word not in FILLER]
        for first, second in zip(solid, solid[1:]):
            pair = "b:" + first + "_" + second
            vector[pair] = vector.get(pair, 0) + \
                (self._weight(first) + self._weight(second)) // 4
        return vector

    # -- compiling: one text -> one key ---------------------------------
    def compile(self, target: str, text: str) -> None:
        self._texts.append((str(target), str(text)))

    def build(self) -> None:
        """Weigh the words over everything compiled, then encode the keys."""
        self.df, self.docs = {}, len(self._texts)
        for _target, text in self._texts:
            for word in set(self.words(text)):
                self.df[word] = self.df.get(word, 0) + 1
        self._vocabulary = sorted(self.df)
        self._gram_cache = {}
        weights = sorted(isqrt((max(self.docs, 1) << 24) // count)
                         for count in self.df.values())
        self._middle = weights[len(weights) // 2] if weights else ONE
        self.keys = []
        for target, text in self._texts:
            vector = self.encode(text)
            norm2 = sum(value * value for value in vector.values())
            if norm2:
                self.keys.append((target, vector, norm2, text))

    def learn(self, target: str, text: str) -> None:
        """Growth: a phrasing someone used for a place becomes a key."""
        self.compile(target, text)
        self.build()

    # -- routing --------------------------------------------------------
    def scores(self, text: str) -> list:
        """[(cosine in fixed point, target)] best key per target, best first;
        ties go to the target that sorts first."""
        query = self.encode(text, mend=True)
        norm2 = sum(value * value for value in query.values())
        best: dict = {}
        if norm2:
            for target, vector, key2, _text in self.keys:
                small, large = (query, vector) if len(query) < len(vector) \
                    else (vector, query)
                dot = sum(value * large[name] for name, value in small.items()
                          if name in large)
                if dot <= 0:
                    continue
                cos = (dot * ONE) // isqrt(norm2 * key2)
                if cos > best.get(target, 0):
                    best[target] = cos
        return sorted(((cos, target) for target, cos in best.items()),
                      key=lambda row: (-row[0], row[1]))

    def determine(self, text: str, *, probable: int, margin: int,
                  possible: int, most: int = 3) -> dict:
        """PROBABLE (one answer), POSSIBLE (a short list to choose from) or
        UNKNOWN (nothing is asserted)."""
        ranked = self.scores(text)
        top = ranked[0][0] if ranked else 0
        second = ranked[1][0] if len(ranked) > 1 else 0
        if ranked and top >= probable and top - second >= margin:
            verdict = "PROBABLE"
        elif ranked and top >= possible:
            verdict = "POSSIBLE"
        else:
            verdict = "UNKNOWN"
        return {"verdict": verdict,
                "value": ranked[0][1] if verdict == "PROBABLE" else None,
                "candidates": [target for _cos, target in ranked[:most]]
                if verdict != "UNKNOWN" else [],
                "score": top, "margin": top - second}

    def state_hash(self) -> str:
        blob = [[target, sorted(vector.items())] for target, vector, _n, _t
                in self.keys]
        return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()
                              ).hexdigest()
