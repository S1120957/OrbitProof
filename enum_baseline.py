"""Enumeration-based evaluation of the order-constrained path query.

This is what an engine does when the time-order condition is written as a
predicate over a variable-length path (e.g. a Cypher `all(...)` over
relationships(p)) and is evaluated over enumerated trails:
  post   : enumerate trails up to H hops, check the order predicate at the end
  pruned : check the predicate incrementally (extend only if k >= last k)
Trails = no repeated contact (relationship-isomorphism).
Both stop after CAP expansions and report the count.
"""
from __future__ import annotations
from collections import defaultdict


def window_adj(ix, k0, k1):
    cp = ix.cp
    lo, hi = ix.slot_start[k0], ix.slot_start[min(k1, cp.T - 1) + 1]
    adj = defaultdict(list)
    for a, b, k in zip(cp.u[lo:hi].tolist(), cp.v[lo:hi].tolist(), cp.k[lo:hi].tolist()):
        adj[a].append((b, k))
    return adj


def enumerate_paths(ix, s, d, k0, k1, H, mode="pruned", cap=10**6):
    adj = window_adj(ix, k0, k1)
    expansions = 0
    found = False
    used = set()
    capped = False

    def dfs(a, depth, last_k, ordered):  # last_k = slot of previous contact
        nonlocal expansions, found, capped
        if capped:
            return
        if a == d and depth > 0 and ordered:
            found = True
        if depth == H:
            return
        for (b, k) in adj.get(a, ()):
            e = (a, b, k)
            if e in used:
                continue
            ok = k >= last_k
            if mode == "pruned" and not ok:
                continue
            expansions += 1
            if expansions >= cap:
                capped = True
                return
            used.add(e)
            dfs(b, depth + 1, k, ordered and ok)
            used.discard(e)

    dfs(s, 0, k0, True)
    return found, expansions, capped
