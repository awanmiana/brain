#!/usr/bin/env python3
"""Generate a synthetic logic/programming-concepts corpus.

Writes `corpus_logic.txt`: logic constructs, algorithms, data structures,
programming concepts and notions for the 1M logic variant.
"""

import random
import sys
from pathlib import Path

TOPICS = [
    ("logic constructs",
     "Logic constructs: if/else branches pick between two paths; loops repeat while a condition holds; "
     "boolean algebra (and/or/not) combines predicates; implications and equivalences relate statements. "
     "Example: 'if not items: return []' — the empty-case guard. Truth tables are how you verify boolean expressions."),
    ("algorithms",
     "Algorithm example — binary search: repeatedly halve the search interval while the target lies inside it; "
     "O(log n) comparisons per search. Dijkstra's algorithm visits the closest unvisited node next until all "
     "nodes are reached — the classic shortest-path algorithm."),
    ("data structures",
     "Data structures: arrays give O(1) random access by index; hash maps give O(1) average lookup by key; "
     "trees organize ordered data (binary search trees, heaps for priority queues); graphs model relations "
     "(nodes + edges, weighted or unweighted, directed or not)."),
    ("complexity",
     "Complexity analysis (big-O): count the dominant operation as n grows. An O(n log n) sort (merge sort) "
     "is fine for large data; an O(n^2) nested loop is the danger sign when n is in the thousands. "
     "Amortized analysis averages work over many operations (e.g., dynamic arrays double capacity when full)."),
    ("recursion",
     "Recursion: a function that calls itself; the recursion terminates via a base case (e.g., 'if n <= 0: return 0'). "
     "Tail recursion and memoization (caching results per input) are the classic patterns — memoization turns "
     "exponential recursive solutions into polynomial ones."),
    ("dynamic programming",
     "Dynamic programming: solve overlapping subproblems once, store answers in a table, and build up from small "
     "to large inputs; example: the longest-common-subsequence table filled with max() over two choices."),
    ("sorting",
     "Sorting: insertion sort is O(n^2) but simple; quicksort is O(n log n) average with a good pivot; merge sort "
     "is O(n log n) worst-case and stable; counting sort works when keys are integers in a small range."),
    ("graphs",
     "Graphs: breadth-first search (queue) finds shortest paths in unweighted graphs; depth-first search "
     "(stack or recursion) explores all reachable nodes; connected components are found by repeating a "
     "traversal from each unvisited node."),
    ("proof",
     "Proof techniques: induction proves statements for all n by showing the base case and that "
     "'if it holds for n then it holds for n+1'; contradiction starts by assuming the opposite and deriving "
     "an impossible result; invariant-based proofs state a fact that holds at every loop iteration."),
    ("set theory",
     "Set theory and relations: a set is an unordered collection of distinct elements; union/intersection/"
     "difference describe combining and filtering; equivalence relations split data into equivalence classes."),
    ("state machines",
     "State machines: states and transitions; a DFA (deterministic) has at most one transition per "
     "(state, symbol) pair — used for pattern matching; FSMs with output are transducers."),
    ("greedy",
     "Greedy algorithms: make the locally optimal choice at each step — classic example: scheduling intervals "
     "by earliest finish time maximizes the number of non-overlapping intervals; sometimes greedy is wrong, "
     "in which case you need DP or backtracking."),
    ("stack frames",
     "Stack frames and the call stack: each function call pushes a frame; deep recursion can overflow the stack; "
     "iterative loops avoid this entirely."),
    ("invariants",
     "Invariants and loop verification: state a fact that is true before/after every loop iteration — if it is, "
     "the loop is correct; e.g., 'the first k elements are sorted' in insertion sort."),
    ("probability",
     "Probability basics: expected value is the average outcome weighted by probabilities; the pigeonhole "
     "principle: n+1 items into n boxes guarantees a shared box."),
    ("concepts",
     "Programming concepts: encapsulation keeps data with its behavior; composition builds systems from parts; "
     "abstraction hides details behind interfaces; decoupling makes changes safer."),
]


def main(out="corpus_logic.txt"):
    rng = random.Random(20261132)
    chunks = []
    topics = list(TOPICS)
    rng.shuffle(topics)
    for _name, text in topics:
        chunks.append(text + "\n\n")
        chunks.append("Explain this topic in terms of its core ideas, example, and when to use it.\n\n")
    text = "".join(chunks)
    Path(out).write_text(text)
    print(f"wrote {out}: {len(text)} chars, {len(TOPICS)} topics")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
