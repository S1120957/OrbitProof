"""Negative (mutation) tests for the certificate checker.

Every mutant below is invalid by construction; the checker must reject it.
Valid certificates are also counted (they must be accepted).
"""
from __future__ import annotations
import random
from collections import Counter
from teg import (INF, earliest_arrival, journey_from_parent, verify_journey, verify_labeling,
                 check_answer, arrival, check_via_answer, check_hops_answer, earliest_arrival_hops,
                 journey_from_parent_hops, loop_eliminate, verify_labeling_hops)

CLASSES = [
    ("phantom", "witness: contact moved to a slot where it does not exist"),
    ("order", "witness: two contacts swapped out of time order"),
    ("chain", "witness: consecutive contacts no longer connected"),
    ("truncate", "witness: last contact removed"),
    ("deadline", "witness checked against a deadline it misses"),
    ("depart", "witness uses a contact before the departure slot"),
    ("avoid", "witness passes through an avoided node"),
    ("hops", "witness exceeds the hop bound"),
    ("f_dest", "labeling: destination label raised"),
    ("f_unreach", "labeling: destination claimed unreachable"),
    ("f_inter", "labeling: intermediate node label raised"),
    ("f_source", "labeling: source label after departure"),
    ("ans_early", "answer: earliest time claimed one slot too early"),
    ("ans_late", "answer: earliest time claimed one slot too late"),
    ("ans_no", "answer: 'no' claimed for a reachable deadline"),
    ("ans_yes", "answer: 'yes' claimed with a journey that misses the deadline"),
    ("ea_forged_lb", "EARLIEST: closed labeling lowered to an unattained claim"),
    ("via_forged_lb", "VIA EARLIEST: second-stage labeling lowered to an unattained claim"),
    ("via_bypass", "VIA: 'yes' claimed with a journey that does not pass the via node"),
    ("hops_forged_lb", "HOPS EARLIEST: hop labeling lowered to an unattained claim"),
]


def run(ix, queries, tau=0, seed=0):
    rng = random.Random(seed)
    tried, rejected = Counter(), Counter()
    valid_ok = valid_n = 0
    component_accepts = Counter()          # forged certificates the component predicates alone accept

    def expect_reject(cls, accepted):
        tried[cls] += 1
        rejected[cls] += (not accepted)

    for (s, d, k0, k1) in queries:
        f, par, _ = earliest_arrival(ix, s, k0, k1, tau=tau)
        if f[d] >= INF:
            valid_n += 1
            valid_ok += check_answer(ix, "EXISTS", s, d, k0, k1, False, None, f, tau=tau)
            fl, pl, _ = earliest_arrival(ix, s, k0, ix.T - 1, tau=tau)
            if fl[d] < INF:
                Jl = journey_from_parent(pl, s, d)
                expect_reject("ans_yes", check_answer(ix, "EXISTS", s, d, k0, k1, True, Jl, None, tau=tau))
            continue
        ks = f[d]
        J = journey_from_parent(par, s, d)
        m = len(J)
        valid_n += 1
        valid_ok += check_answer(ix, "EARLIEST", s, d, k0, k1, ks, J, f, tau=tau)

        # --- witness mutations
        i = rng.randrange(m)
        a, b, k = J[i]
        cands = [x for x in range(k0, ix.T) if (a, b, x) not in ix.cset]
        if cands:
            lo = J[i - 1][2] + tau if i > 0 else k0
            hi = J[i + 1][2] - tau if i + 1 < m else k1 - tau
            pref = [x for x in cands if lo <= x <= hi] or cands
            Jm = J[:i] + [(a, b, rng.choice(pref))] + J[i + 1:]
            expect_reject("phantom", verify_journey(ix, Jm, s, d, k0, ix.T - 1, tau=tau))
        pairs = [j for j in range(m - 1) if J[j][2] != J[j + 1][2]]
        if pairs:
            j = rng.choice(pairs)
            (a1, b1, x1), (a2, b2, x2) = J[j], J[j + 1]
            Jm = J[:j] + [(a1, b1, x2), (a2, b2, x1)] + J[j + 2:]
            expect_reject("order", verify_journey(ix, Jm, s, d, k0, ix.T - 1, tau=tau))
        if m >= 2:
            j = rng.randrange(m - 1)
            a1, b1, x1 = J[j]
            other = rng.choice([n for n in range(ix.N) if n not in (b1, a1)])
            Jm = J[:j] + [(a1, other, x1)] + J[j + 1:]
            expect_reject("chain", verify_journey(ix, Jm, s, d, k0, ix.T - 1, tau=tau))
            mid = rng.choice([c[1] for c in J[:-1]])
            expect_reject("avoid", verify_journey(ix, J, s, d, k0, k1, avoid=frozenset([mid]), tau=tau))
            expect_reject("hops", verify_journey(ix, J, s, d, k0, k1, H=m - 1, tau=tau))
        expect_reject("truncate", verify_journey(ix, J[:-1], s, d, k0, k1, tau=tau))
        arr = arrival(J, k0, tau)
        expect_reject("deadline", verify_journey(ix, J, s, d, k0, arr - 1, tau=tau))
        expect_reject("depart", verify_journey(ix, J, s, d, J[0][2] + 1, k1, tau=tau))

        # --- labeling mutations (window [k0, k*])
        g = list(f); g[d] = ks + 1
        expect_reject("f_dest", verify_labeling(ix, g, s, k0, ks, tau=tau))
        g = list(f); g[d] = INF
        expect_reject("f_unreach", verify_labeling(ix, g, s, k0, ks, tau=tau))
        inter = [c[1] for c in J[:-1]]
        if inter:
            v = rng.choice(inter)
            g = list(f); g[v] = f[v] + 1
            expect_reject("f_inter", verify_labeling(ix, g, s, k0, ks, tau=tau))
        g = list(f); g[s] = k0 + 1
        expect_reject("f_source", verify_labeling(ix, g, s, k0, ks, tau=tau))

        # --- answer-level mutations
        if ks > k0:
            expect_reject("ans_early", check_answer(ix, "EARLIEST", s, d, k0, k1, ks - 1, J, f, tau=tau))
        expect_reject("ans_late", check_answer(ix, "EARLIEST", s, d, k0, k1, ks + 1, J, f, tau=tau))
        expect_reject("ans_no", check_answer(ix, "EXISTS", s, d, k0, ks, False, None, f, tau=tau))
        if ks > k0:
            forged = [min(x, ks - 1) for x in f]
            component_accepts["ea_forged_lb"] += verify_labeling(ix, forged, s, k0, k1, tau=tau) and \
                verify_journey(ix, J, s, d, k0, k1, tau=tau)
            expect_reject("ea_forged_lb", check_answer(ix, "EARLIEST", s, d, k0, k1, ks - 1, J, forged, tau=tau))

        # --- VIA (a ground station other than s and d)
        g = rng.choice([n for n in range(ix.N) if ix.cp.kinds[n] == "GS" and n not in (s, d)])
        if f[g] < INF:
            kg = f[g]
            f2, p2, _ = earliest_arrival(ix, g, kg, k1, tau=tau)
            if f2[d] < INF:
                Jv = journey_from_parent(par, s, g) + journey_from_parent(p2, g, d)
                kv = f2[d]
                valid_n += 1
                valid_ok += check_via_answer(ix, "EARLIEST", s, d, g, k0, k1, kv, Jv, f, f2, tau=tau)
                if kv - 1 >= kg:
                    forged = [min(x, kv - 1) for x in f2]
                    component_accepts["via_forged_lb"] += verify_labeling(ix, forged, g, kg, k1, tau=tau) and \
                        verify_journey(ix, Jv, s, d, k0, k1, via=g, tau=tau)
                    expect_reject("via_forged_lb",
                                  check_via_answer(ix, "EARLIEST", s, d, g, k0, k1, kv - 1, Jv, f, forged, tau=tau))
            if g not in {c[1] for c in J[:-1]}:
                expect_reject("via_bypass", check_via_answer(ix, "EXISTS", s, d, g, k0, k1, True, J, f, None, tau=tau))

        # --- HOPS (tau = 0 only)
        if tau == 0:
            H = 3
            fh, ph, _ = earliest_arrival_hops(ix, s, k0, k1, H)
            kh = min(fh[d])
            if kh < INF:
                Jh = loop_eliminate(journey_from_parent_hops(ph, fh, s, d, H), s)
                valid_n += 1
                valid_ok += check_hops_answer(ix, "EARLIEST", s, d, k0, k1, H, kh, Jh, fh)
                if kh > k0:
                    forged = [[min(x, kh - 1) for x in row] for row in fh]
                    component_accepts["hops_forged_lb"] += verify_labeling_hops(ix, forged, s, k0, k1, H) and \
                        verify_journey(ix, Jh, s, d, k0, k1, H=H)
                    expect_reject("hops_forged_lb",
                                  check_hops_answer(ix, "EARLIEST", s, d, k0, k1, H, kh - 1, Jh, forged))

    return dict(tried=dict(tried), rejected=dict(rejected), valid_n=valid_n, valid_ok=valid_ok,
                component_accepts=dict(component_accepts))


if __name__ == "__main__":
    from constellation import make_contact_plan
    from teg import Index
    rng = random.Random(5)
    for reg in ("isl_full", "isl_intra", "no_isl"):
        cp = make_contact_plan(6, 11, 180, reg); ix = Index(cp)
        gs = [i for i, t in enumerate(cp.kinds) if t == "GS"]
        qs = []
        for _ in range(200):
            s, d = rng.sample(gs, 2); k0 = rng.randint(0, 110); qs.append((s, d, k0, k0 + 60))
        for tau in (0, 1):
            r = run(ix, qs, tau)
            print(reg, "tau", tau, "valid accepted", r["valid_ok"], "/", r["valid_n"],
                  "| mutants rejected", sum(r["rejected"].values()), "/", sum(r["tried"].values()),
                  "| forged lower bounds accepted by component predicates alone:", r["component_accepts"])
            assert r["valid_ok"] == r["valid_n"] and sum(r["rejected"].values()) == sum(r["tried"].values())
