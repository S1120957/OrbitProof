"""Time-expanded-graph (TEG) evaluation, lifted witnesses and certificates.

A journey is a list of contacts (u, v, k) with u_0 = s, consecutive endpoints
matching, u_m = d and k0 <= k_1 <= ... <= k_m (store-and-forward between
slots, multi-hop within a slot).

Everything here is engine-independent Python. sqlview.py evaluates the same
queries through a materialised relational view in DuckDB and is used to
cross-check the answers.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict, deque

INF = 10**9


class Index:
    """Per-slot CSR adjacency + hash set of contacts for O(1) membership."""

    def __init__(self, cp):
        self.cp = cp
        self.N, self.T = cp.N, cp.T
        self.adj = [defaultdict(list) for _ in range(cp.T)]
        for a, b, k in zip(cp.u.tolist(), cp.v.tolist(), cp.k.tolist()):
            self.adj[k][a].append(b)
        self.cset = set(zip(cp.u.tolist(), cp.v.tolist(), cp.k.tolist()))
        # contacts sorted by slot for window scans
        self.slot_start = np.searchsorted(cp.k, np.arange(cp.T + 1))


# ----------------------------------------------------------------------------
# Query evaluation on the TEG (sweep over slots; BFS inside each slot)
# ----------------------------------------------------------------------------
def earliest_arrival(ix, s, k0, k1, avoid=frozenset(), tau=0, stop_at=None):
    """Returns (f, parent, visited_states).
    f[n]      : earliest slot from which n holds the bundle (INF if not by k1)
    parent[n] : contact (u, n, k) through which n was first reached
    Reachability of (n, k) from (s, k0) in the TEG view with hold edges
    (n,k)->(n,k+1) and transmission edges (u,k)->(v,k+tau), tau in {0,1}:
      tau = 0 : zero-duration slotted abstraction (multi-hop within a slot)
      tau = 1 : every transmission occupies one slot (one hop per slot)
    stop_at=d stops as soon as f[d] is final (earliest-arrival early exit);
    the labeling is then closed on the contacts with k + tau <= f[d]."""
    N = ix.N
    f = [INF] * N
    parent = [None] * N
    if s in avoid:
        return f, parent, 0
    f[s] = k0
    reached = [s]
    pending = defaultdict(list)
    visited = 0
    kmax = min(k1, ix.T - 1)
    for k in range(k0, kmax + 1):
        if tau:
            reached.extend(pending.pop(k, ()))
        adj = ix.adj[k]
        if tau == 0:
            queue = deque(reached)          # FIFO: fewest extra hops inside the slot
            while queue:
                a = queue.popleft()
                visited += 1
                for b in adj.get(a, ()):
                    if f[b] == INF and b not in avoid:
                        f[b] = k
                        parent[b] = (a, b, k)
                        reached.append(b)
                        queue.append(b)
        else:
            t = k + tau
            for a in reached:
                visited += 1
                if t > kmax:
                    continue
                for b in adj.get(a, ()):
                    if f[b] == INF and b not in avoid:
                        f[b] = t
                        parent[b] = (a, b, k)
                        pending[t].append(b)
        if stop_at is not None and f[stop_at] != INF:
            break
    return f, parent, visited


def arrival(J, k0, tau=0):
    return (J[-1][2] + tau) if J else k0


def earliest_arrival_hops(ix, s, k0, k1, H, avoid=frozenset()):
    """Hop-bounded variant: states (n, h), h = #transmissions so far (arity-3 view)."""
    N = ix.N
    f = [[INF] * (H + 1) for _ in range(N)]
    parent = {}
    f[s][0] = k0
    reached = [(s, 0)]
    visited = 0
    for k in range(k0, min(k1, ix.T - 1) + 1):
        adj = ix.adj[k]
        stack = list(reached)
        while stack:
            a, h = stack.pop()
            visited += 1
            if h == H:
                continue
            for b in adj.get(a, ()):
                if b in avoid:
                    continue
                if f[b][h + 1] == INF:
                    f[b][h + 1] = k
                    parent[(b, h + 1)] = (a, h, k)
                    reached.append((b, h + 1))
                    stack.append((b, h + 1))
    return f, parent, visited


def journey_from_parent(parent, s, d):
    J = []
    n = d
    while n != s:
        a, b, k = parent[n]
        J.append((a, b, k))
        n = a
    return J[::-1]


def journey_from_parent_hops(parent, f, s, d, H):
    best = min(range(H + 1), key=lambda h: (f[d][h], h))
    if f[d][best] >= INF:
        return None
    J, n, h = [], d, best
    while (n, h) != (s, 0):
        a, ha, k = parent[(n, h)]
        J.append((a, n, k))
        n, h = a, ha
    return J[::-1]


def loop_eliminate(J, s):
    """Lemma 2: remove cycles (wait instead of looping); O(m)."""
    out = []
    pos = {s: 0}
    for c in J:
        a, b, k = c
        out.append(c)
        if b in pos:                       # b visited before: cut the loop
            cut = pos[b]
            del out[cut:]
            pos = {s: 0}
            for i, (x, y, _) in enumerate(out):
                pos[y] = i + 1
        else:
            pos[b] = len(out)
    return out


def is_node_simple(J, s):
    nodes = [s] + [c[1] for c in J]
    return len(nodes) == len(set(nodes))


# ----------------------------------------------------------------------------
# Independent verifiers (the "trusted checker")
# ----------------------------------------------------------------------------
def verify_journey(ix, J, s, d, k0, k1, avoid=frozenset(), via=None, H=None, tau=0):
    """O(m): witness is a real, time-respecting journey satisfying the query."""
    if s == d and not J:
        return True
    if not J or J[0][0] != s or J[-1][1] != d:
        return False
    avail = k0                      # slot from which the bundle is available at the current node
    for i, (a, b, k) in enumerate(J):
        if (a, b, k) not in ix.cset:
            return False
        if i > 0 and J[i - 1][1] != a:
            return False
        if k < avail:
            return False
        if a in avoid or b in avoid:
            return False
        avail = k + tau
    if avail > k1:
        return False
    if via is not None and via not in {c[1] for c in J[:-1]} | {s}:
        return False
    if H is not None and len(J) > H:
        return False
    return True


def verify_labeling(ix, f, s, k0, k1, avoid=frozenset(), tau=0):
    """Proposition 3: (C1) f[s] <= k0 and (C2) f[u] <= k  =>  f[v] <= k + tau for every
    contact (u, v, k) with k0 <= k and k + tau <= k1. If True, every journey that reaches
    n at slot k <= k1 has k >= f[n]. Cost: one scan of the contacts in the window."""
    if f[s] > k0:
        return False
    cp = ix.cp
    hi_k = min(k1 - tau, cp.T - 1)
    if hi_k < k0:
        return True
    lo, hi = ix.slot_start[k0], ix.slot_start[hi_k + 1]
    for a, b, k in zip(cp.u[lo:hi].tolist(), cp.v[lo:hi].tolist(), cp.k[lo:hi].tolist()):
        if a in avoid or b in avoid:
            continue
        if f[a] <= k and f[b] > k + tau:
            return False
    return True


def check_answer(ix, task, s, d, k0, k1, claim, J, f, avoid=frozenset(), tau=0):
    """Answer-level checker used by the agent.
    EXISTS  yes  : J is a valid journey arriving by k1
    EXISTS  no   : f closed on [k0, k1] and f[d] > k1
    EARLIEST k*  : J valid with arrival == k*, and f closed on [k0, k*-1] with f[d] >= k*
    EARLIEST none: f closed on [k0, k1] and f[d] > k1"""
    if task == "EXISTS":
        if claim:
            return J is not None and verify_journey(ix, J, s, d, k0, k1, avoid, tau=tau)
        return f is not None and f[d] > k1 and verify_labeling(ix, f, s, k0, k1, avoid, tau)
    if claim is None:
        return f is not None and f[d] > k1 and verify_labeling(ix, f, s, k0, k1, avoid, tau)
    if J is None or f is None:
        return False
    return (verify_journey(ix, J, s, d, k0, claim, avoid, tau=tau) and arrival(J, k0, tau) == claim
            and f[d] >= claim and verify_labeling(ix, f, s, k0, claim - 1, avoid, tau))


def check_via_answer(ix, task, s, d, g, k0, k1, claim, J, f1, f2, avoid=frozenset(), tau=0):
    """Answer-level checker for VIA(g). f1: labeling from (s, k0); f2: labeling from (g, f1[g]).
    A labeling only certifies a LOWER bound; an earliest answer also needs a journey that
    ATTAINS the claimed value (arrival(J) == claim) and passes g."""
    if f1 is None or not verify_labeling(ix, f1, s, k0, k1, avoid, tau):
        return False
    kg = f1[g]                                   # certified lower bound on reaching g
    if task == "EXISTS" and claim:
        return J is not None and verify_journey(ix, J, s, d, k0, k1, avoid, via=g, tau=tau)
    if (task == "EXISTS" and not claim) or (task == "EARLIEST" and claim is None):
        if kg > k1:
            return True
        return f2 is not None and f2[d] > k1 and verify_labeling(ix, f2, g, kg, k1, avoid, tau)
    if J is None or f2 is None or kg > claim:
        return False
    return (verify_journey(ix, J, s, d, k0, claim, avoid, via=g, tau=tau)
            and arrival(J, k0, tau) == claim                          # attainment
            and f2[d] >= claim and verify_labeling(ix, f2, g, kg, claim - 1, avoid, tau))


def check_hops_answer(ix, task, s, d, k0, k1, H, claim, J, f, avoid=frozenset()):
    """Answer-level checker for HOPS(H) (tau = 0); f is the hop-layered labeling f[n][h]."""
    if task == "EXISTS" and claim:
        return J is not None and verify_journey(ix, J, s, d, k0, k1, avoid, H=H)
    if f is None:
        return False
    if (task == "EXISTS" and not claim) or (task == "EARLIEST" and claim is None):
        return min(f[d]) > k1 and verify_labeling_hops(ix, f, s, k0, k1, H, avoid)
    if J is None:
        return False
    return (verify_journey(ix, J, s, d, k0, claim, avoid, H=H)
            and arrival(J, k0) == claim                               # attainment
            and min(f[d]) >= claim and verify_labeling_hops(ix, f, s, k0, claim - 1, H, avoid))


def verify_labeling_hops(ix, f, s, k0, k1, H, avoid=frozenset()):
    """Hop-layered analogue of verify_labeling on states (n, h), h <= H."""
    if f[s][0] > k0:
        return False
    cp = ix.cp
    if min(k1, cp.T - 1) < k0:
        return True
    lo, hi = ix.slot_start[k0], ix.slot_start[min(k1, cp.T - 1) + 1]
    for a, b, k in zip(cp.u[lo:hi].tolist(), cp.v[lo:hi].tolist(), cp.k[lo:hi].tolist()):
        if a in avoid or b in avoid:
            continue
        fa, fb = f[a], f[b]
        for h in range(H):
            if fa[h] <= k and fb[h + 1] > k:
                return False
    return True


# ----------------------------------------------------------------------------
# Replay on the lifted witness (sub-instance) and minimality checks
# ----------------------------------------------------------------------------
def replay_earliest(ix, J, s, d, k0, drop=None, tau=0):
    """Earliest arrival at d when the query is replayed on B + W_C(J), where the
    background B (Node, Slot, Succ) is fixed and W_C(J) are the witness contacts.
    drop: one contact of J to remove (minimality check)."""
    contacts = [c for c in J if c != drop]
    adj = defaultdict(lambda: defaultdict(list))
    for a, b, k in contacts:
        adj[k][a].append(b)
    f = [INF] * ix.N
    f[s] = k0
    reached = [s]
    pending = defaultdict(list)
    for k in range(k0, ix.T):
        if tau:
            reached.extend(pending.pop(k, ()))
            for a in list(reached):
                for b in adj[k].get(a, ()):
                    if f[b] == INF and k + tau < ix.T:
                        f[b] = k + tau
                        pending[k + tau].append(b)
        else:
            stack = list(reached)
            while stack:
                a = stack.pop()
                for b in adj[k].get(a, ()):
                    if f[b] == INF:
                        f[b] = k
                        reached.append(b)
                        stack.append(b)
    return f[d]


def witness_is_minimal(ix, J, s, d, k0, tau=0):
    """Proposition 2: removing any witness contact makes 'arrival <= arr(J)' false."""
    arr = arrival(J, k0, tau)
    return all(replay_earliest(ix, J, s, d, k0, drop=c, tau=tau) > arr for c in J)


def coarse_witness_size(ix, J, s, k0):
    """Coarse subgraph: every contact in [k0, arr] incident to a node on the journey."""
    nodes = {s} | {c[1] for c in J}
    arr = J[-1][2]
    cp = ix.cp
    lo, hi = ix.slot_start[k0], ix.slot_start[arr + 1]
    uu, vv = cp.u[lo:hi], cp.v[lo:hi]
    mask = np.isin(uu, list(nodes)) | np.isin(vv, list(nodes))
    return int(mask.sum())


# ----------------------------------------------------------------------------
# Time-agnostic semantics that a naive text-to-query translation may produce
# ----------------------------------------------------------------------------
def static_reach(ix, s, d, k0, k1):
    """Path in the union of all contacts in [k0,k1], ignoring their order."""
    cp = ix.cp
    lo, hi = ix.slot_start[k0], ix.slot_start[min(k1, cp.T - 1) + 1]
    adj = defaultdict(set)
    for a, b in zip(cp.u[lo:hi].tolist(), cp.v[lo:hi].tolist()):
        adj[a].add(b)
    seen, stack = {s}, [s]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in seen:
                seen.add(b); stack.append(b)
    return d in seen


def snapshot_reach(ix, s, d, k0):
    """Path in the single snapshot at k0 (no store-and-forward)."""
    adj = ix.adj[k0]
    seen, stack = {s}, [s]
    while stack:
        a = stack.pop()
        for b in adj.get(a, ()):
            if b not in seen:
                seen.add(b); stack.append(b)
    return d in seen
