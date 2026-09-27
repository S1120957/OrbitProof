"""Headline numbers of Tables III and IV under the 60-s default and under 10-s slots,
to support the decision on the default discretisation. Writes alt_slot_width.md.
Physical windows (15/30/60 min) and query counts are identical across slot widths."""
from __future__ import annotations
import random, statistics as st
from constellation import make_contact_plan
from teg import (Index, INF, earliest_arrival, journey_from_parent, coarse_witness_size,
                 static_reach, snapshot_reach, check_answer)

NQ = 500
REGS = ["isl_full", "isl_intra", "no_isl"]


def run(dt, rng_seed=17):
    rng = random.Random(rng_seed)
    out = {}
    for reg in REGS:
        T = int(180 * 60 / dt)
        cp = make_contact_plan(6, 11, T, reg, dt=dt)
        ix = Index(cp)
        gs = [i for i, t in enumerate(cp.kinds) if t == "GS"]
        per = 60 / dt                                     # slots per minute
        R = dict(contacts=cp.n_contacts)
        L, red, delay, reach, certified = [], [], [], 0, 0
        for _ in range(NQ):
            s, d = rng.sample(gs, 2); t0 = rng.uniform(0, 119)
            k0 = int(t0 * per); k1 = min(k0 + int(60 * per), T - 1)
            f, par, _ = earliest_arrival(ix, s, k0, k1)
            if f[d] >= INF:
                certified += check_answer(ix, "EXISTS", s, d, k0, k1, False, None, f)
                continue
            reach += 1
            J = journey_from_parent(par, s, d)
            certified += check_answer(ix, "EARLIEST", s, d, k0, k1, f[d], J, f)
            L.append(len(J)); cw = coarse_witness_size(ix, J, s, k0); red.append(1 - len(J) / cw)
            delay.append((f[d] - k0) / per)
        R.update(reach=reach, L_med=st.median(L), L_max=max(L), red_med=st.median(red),
                 delay_med=st.median(delay), certified=certified)
        for Wm in (15, 30, 60):
            es = en = tr = 0
            for _ in range(NQ):
                s, d = rng.sample(gs, 2); t0 = rng.uniform(0, 119)
                k0 = int(t0 * per); k1 = min(k0 + int(Wm * per), T - 1)
                f, _, _ = earliest_arrival(ix, s, k0, k1)
                truth = f[d] < INF; tr += truth
                es += static_reach(ix, s, d, k0, k1) != truth
                en += snapshot_reach(ix, s, d, k0) != truth
            R[f"sem{Wm}"] = (100 * tr / NQ, 100 * es / NQ, 100 * en / NQ)
        out[reg] = R
        print(dt, reg, R, flush=True)
    return out


def main():
    res = {dt: run(dt) for dt in (60, 10)}
    name = {"isl_full": "Full ISLs", "isl_intra": "Intra-plane", "no_isl": "No ISLs"}
    lines = ["# Default slot width: 60 s vs 10 s (tau = 0)", "",
             f"{NQ} queries per regime and window; same physical windows; seed 17.", "",
             "## Witness statistics (60-min window)", "",
             "| Regime | Slot | Contacts | Reachable | Median delay [min] | Witness median/max | Size reduction | Certified |",
             "|---|---|---|---|---|---|---|---|"]
    for reg in REGS:
        for dt in (60, 10):
            R = res[dt][reg]
            lines.append(f"| {name[reg]} | {dt} s | {R['contacts']:,} | {R['reach']}/{NQ} | {R['delay_med']:.1f} | "
                         f"{R['L_med']:.0f}/{R['L_max']} | {100*R['red_med']:.1f}% | {R['certified']}/{NQ} |")
    lines += ["", "## Error of time-agnostic semantics (reachable / static error / snapshot error, %)", "",
              "| Regime | Slot | 15 min | 30 min | 60 min |", "|---|---|---|---|---|"]
    for reg in REGS:
        for dt in (60, 10):
            R = res[dt][reg]
            cells = [f"{R[f'sem{W}'][0]:.1f} / {R[f'sem{W}'][1]:.1f} / {R[f'sem{W}'][2]:.1f}" for W in (15, 30, 60)]
            lines.append(f"| {name[reg]} | {dt} s | " + " | ".join(cells) + " |")
    open("alt_slot_width.md", "w").write("\n".join(lines) + "\n")
    import os
    ten = res[10]
    m = {"AltStaticNoIslMin": min(ten["no_isl"][f"sem{W}"][1] for W in (15, 30, 60)),
         "AltStaticNoIslMax": max(ten["no_isl"][f"sem{W}"][1] for W in (15, 30, 60)),
         "AltSnapIslMin": min(ten[r][f"sem{W}"][2] for r in ("isl_full", "isl_intra") for W in (15, 30, 60)),
         "AltSnapIslMax": max(ten[r][f"sem{W}"][2] for r in ("isl_full", "isl_intra") for W in (15, 30, 60))}
    gen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "gen")
    with open(os.path.join(gen, "alt_macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\R{k}}}{{{v:.0f}}}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
