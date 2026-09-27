"""Independent oracle on small random contact plans.

The oracle enumerates journeys directly from Definition 1 (contacts in non-decreasing
slot order, departure >= k0), with no code shared with the engine. For every random
query (EXISTS / EARLIEST, optional AVOID, VIA or HOPS) the engine answer must equal the
oracle answer and its exported certificate must be accepted. Exits non-zero on mismatch."""
import random, sys
import numpy as np
from constellation import ContactPlan
from teg import Index
from agent import JourneyIR, execute, verify_certificate, slot_to_hhmm


def oracle(contacts, s, d, k0, k1, avoid, via, H):
    """Min arrival over all journeys (walks) of length <= limit satisfying the query, or None."""
    by_src = {}
    for (u, v, k) in contacts:
        by_src.setdefault(u, []).append((v, k))
    limit = H if H is not None else 2 * len({x for c in contacts for x in c[:2]}) + 2
    best = [None]

    def dfs(node, avail, depth, seen_via):
        if node in avoid:
            return
        if node == d and (via is None or seen_via) and avail <= k1:
            best[0] = avail if best[0] is None else min(best[0], avail)
        if depth == limit:
            return
        for (v, k) in by_src.get(node, ()):
            if avail <= k <= k1 and v not in avoid:
                dfs(v, k, depth + 1, seen_via or v == via)

    if s not in avoid:
        dfs(s, k0, 0, via is None or s == via)
    return best[0]


def main(n_plans=300, n_queries=25, seed=0):
    rng = random.Random(seed)
    checked = mismatches = uncertified = 0
    for _ in range(n_plans):
        N, T = rng.randint(3, 6), rng.randint(3, 7)
        names = [f"N{i}" for i in range(N)]
        m = rng.randint(0, 3 * N)
        cs = set()
        for _ in range(m):
            u, v = rng.sample(range(N), 2)
            cs.add((u, v, rng.randrange(T)))
        cs = sorted(cs, key=lambda c: (c[2], c[0], c[1])) if rng.random() < 0.5 else list(cs)  # also unsorted
        arr = np.array(cs, dtype=np.int32).reshape(-1, 3)
        ix = Index(ContactPlan(names, ["GS"] * N, T, arr[:, 0], arr[:, 1], arr[:, 2]))
        for _ in range(n_queries):
            s, d = rng.randrange(N), rng.randrange(N)
            k0 = rng.randrange(T); k1 = rng.randint(k0, T - 1)
            task = rng.choice(["EXISTS", "EARLIEST"])
            kind = rng.choice(["plain", "avoid", "via", "hops"])
            avoid, via, H = [], None, None
            if kind == "avoid":
                avoid = [names[x] for x in rng.sample(range(N), 1) if x not in (s, d)]
            elif kind == "via":
                via = names[rng.randrange(N)]
            elif kind == "hops":
                H = rng.randint(1, 3)
            ir = JourneyIR(task, names[s], names[d], slot_to_hhmm(k0),
                           slot_to_hhmm(k1) if (task == "EXISTS" or rng.random() < 0.3) else None,
                           avoid, via, H)
            kk1 = k1 if ir.arrive_by is not None else T - 1
            truth = oracle(cs, s, d, k0, kk1, {names.index(a) for a in avoid},
                           names.index(via) if via else None, H)
            want = (truth is not None) if task == "EXISTS" else truth
            res = execute(ix, names, ir)
            checked += 1
            if res.answer != want or (type(res.answer) is bool) != (task == "EXISTS"):
                mismatches += 1
                print("MISMATCH", cs, ir, "engine", res.answer, "oracle", want)
            if not (res.certified and verify_certificate(ix, names, res.detail["certificate"])):
                uncertified += 1
                print("UNCERTIFIED", cs, ir, res)
    print(f"{checked} queries on {n_plans} random plans: {mismatches} answer mismatches, "
          f"{uncertified} answers without an accepted certificate")
    import os
    gen = os.environ.get("ORBITPROOF_GEN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "gen"))
    if mismatches == 0 and uncertified == 0 and os.path.isdir(gen):
        with open(os.path.join(gen, "oracle_macros.tex"), "w") as fh:
            fh.write(f"\\newcommand{{\\RORq}}{{{checked:,}}}\n".replace(",", "{,}"))
            fh.write(f"\\newcommand{{\\RORplans}}{{{n_plans}}}\n")
    return mismatches == 0 and uncertified == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
