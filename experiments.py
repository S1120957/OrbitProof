"""Runs every experiment and writes out/figs/*.pdf, out/gen/*.tex and out/results.json
   (output directory: $ORBITPROOF_OUT, default ./out).
Usage:  python3 experiments.py            (full run, ~5-10 min on one core)
        python3 experiments.py --quick    (smoke test)
"""
from __future__ import annotations
import argparse, json, os, random, statistics as st, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constellation import make_contact_plan
from teg import (Index, INF, earliest_arrival, earliest_arrival_hops, journey_from_parent,
                 verify_journey, verify_labeling, replay_earliest, witness_is_minimal,
                 is_node_simple, coarse_witness_size, static_reach, snapshot_reach)
from enum_baseline import enumerate_paths
from sqlview import SQLView
from teg import check_answer, arrival
import mutation, gc, platform

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("ORBITPROOF_OUT", os.path.join(HERE, "out"))   # output directory (not tracked)
T_SLOTS = 180
SEED = 7

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, not Type 3 (IEEE PDF eXpress)
                     "font.size": 7, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.linewidth": 0.5, "lines.linewidth": 1.0, "lines.markersize": 3,
                     "legend.fontsize": 6, "legend.frameon": False,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5})


def gs_ids(cp):
    return [i for i, t in enumerate(cp.kinds) if t == "GS"]


def rand_queries(cp, n, W, rng):
    gs = gs_ids(cp)
    out = []
    for _ in range(n):
        s, d = rng.sample(gs, 2)
        k0 = rng.randint(0, cp.T - W - 1)
        out.append((s, d, k0, k0 + W))
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


# ---------------------------------------------------------------------------
def exp_scalability(quick, rng):
    sizes = [(6, 11), (12, 12)] if quick else [(6, 11), (12, 12), (12, 22), (24, 22)]
    nq = 5 if quick else 40
    rows = []
    for (P, S) in sizes:
        t = time.perf_counter()
        cp = make_contact_plan(P, S, T_SLOTS, "isl_full")
        gen_s = time.perf_counter() - t
        ix = Index(cp)
        sv = SQLView(cp)
        qs = rand_queries(cp, nq, 60, rng)
        tpy, tsql, agree = [], [], 0
        for (s, d, k0, k1) in qs:
            t = time.perf_counter(); f, _, _ = earliest_arrival(ix, s, k0, k1); tpy.append(time.perf_counter() - t)
            t = time.perf_counter(); r = sv.earliest(s, d, k0, k1); tsql.append(time.perf_counter() - t)
            agree += ((f[d] if f[d] < INF else None) == r)
        rows.append(dict(P=P, S=S, sats=P * S, nodes=cp.N, contacts=cp.n_contacts, gen_s=gen_s,
                         vnode=sv.n_vnode, vedge=sv.n_vedge, mat_ms=1000 * sv.materialise_s,
                         py_ms=1000 * med(tpy), sql_ms=1000 * med(tsql), agree=agree, nq=len(qs),
                         py_q=[1000 * pct(tpy, q) for q in (25, 75, 95)],
                         sql_q=[1000 * pct(tsql, q) for q in (25, 75, 95)]))
        print("scal", rows[-1], flush=True)
    return rows


def exp_enumeration(quick, rng):
    cp = make_contact_plan(6, 11, T_SLOTS, "isl_full")
    ix = Index(cp)
    Hs = [1, 2, 3, 4] if quick else [1, 2, 3, 4, 5, 6]
    nq = 4 if quick else 12
    cap = 2 * 10**5 if quick else 10**6
    qs = rand_queries(cp, nq, 30, rng)
    rows = []
    for H in Hs:
        post, pruned, tegv, capped_post, capped_pr = [], [], [], 0, 0
        for (s, d, k0, k1) in qs:
            _, e1, c1 = enumerate_paths(ix, s, d, k0, k1, H, "post", cap=cap)
            _, e2, c2 = enumerate_paths(ix, s, d, k0, k1, H, "pruned", cap=cap)
            _, _, v = earliest_arrival_hops(ix, s, k0, k1, H)
            post.append(e1); pruned.append(e2); tegv.append(v)
            capped_post += c1; capped_pr += c2
        rows.append(dict(H=H, post=med(post), pruned=med(pruned), teg=med(tegv),
                         capped_post=capped_post, capped_pruned=capped_pr, nq=nq, cap=cap))
        print("enum", rows[-1], flush=True)
    return rows


def exp_witness_and_semantics(quick, rng):
    regimes = ["isl_full", "isl_intra", "no_isl"]
    nq = 60 if quick else 500
    n_min = 20 if quick else 100
    out = {}
    for reg in regimes:
        cp = make_contact_plan(6, 11, T_SLOTS, reg)
        ix = Index(cp)
        R = dict(contacts=cp.n_contacts)
        # --- witnesses and certificates (W = 60 slots)
        qs = rand_queries(cp, nq, 60, rng)
        L, coarse, red, tv_j, tv_c, replay_ok, simple_ok, verified, minimal_ok, n_reach = \
            [], [], [], [], [], 0, 0, 0, 0, 0
        wait_frac = []
        for i, (s, d, k0, k1) in enumerate(qs):
            f, par, _ = earliest_arrival(ix, s, k0, k1)
            t = time.perf_counter(); okc = verify_labeling(ix, f, s, k0, k1); tv_c.append(time.perf_counter() - t)
            if f[d] >= INF:
                verified += okc
                continue
            n_reach += 1
            J = journey_from_parent(par, s, d)
            t = time.perf_counter(); okj = verify_journey(ix, J, s, d, k0, k1); tv_j.append(time.perf_counter() - t)
            verified += (okj and okc)
            L.append(len(J))
            cw = coarse_witness_size(ix, J, s, k0)
            coarse.append(cw); red.append(1 - len(J) / cw)
            replay_ok += (replay_earliest(ix, J, s, d, k0) == f[d])
            simple_ok += is_node_simple(J, s)
            if n_reach <= n_min:
                minimal_ok += witness_is_minimal(ix, J, s, d, k0)
            wait_frac.append((f[d] - k0))
        R.update(nq=nq, reach=n_reach, L_med=med(L), L_max=max(L) if L else 0,
                 coarse_med=med(coarse), red_med=med(red), delay_med=med(wait_frac),
                 tj_us=1e6 * med(tv_j), tc_ms=1e3 * med(tv_c), replay_ok=replay_ok,
                 simple_ok=simple_ok, verified=verified, minimal_ok=minimal_ok,
                 minimal_n=min(n_min, n_reach))
        # --- time-agnostic semantics
        sem = {}
        for W in (15, 30, 60):
            qs = rand_queries(cp, nq, W, rng)
            truth, fp_static, fn_snap, err_static, err_snap, pos_static = 0, 0, 0, 0, 0, 0
            for (s, d, k0, k1) in qs:
                f, _, _ = earliest_arrival(ix, s, k0, k1)
                tr = f[d] < INF
                a_st = static_reach(ix, s, d, k0, k1)
                a_sn = snapshot_reach(ix, s, d, k0)
                truth += tr; pos_static += a_st
                err_static += (a_st != tr); err_snap += (a_sn != tr)
                fp_static += (a_st and not tr); fn_snap += (tr and not a_sn)
            sem[W] = dict(truth=truth / nq, err_static=err_static / nq, err_snap=err_snap / nq,
                          fp_static=fp_static / nq, fn_snap=fn_snap / nq)
        R["sem"] = sem
        out[reg] = R
        print("wit/sem", reg, R, flush=True)
    return out


# ---------------------------------------------------------------------------
def exp_horizon(quick, rng):
    """View size / materialisation / memory / query / checker cost vs. time horizon."""
    P, S = (6, 11) if quick else (12, 22)
    hours = [3, 6] if quick else [3, 6, 12, 24]
    nq, nq_sql = (10, 3) if quick else (100, 20)
    rows = []
    for h in hours:
        T = h * 60
        cp = make_contact_plan(P, S, T, "isl_full")
        ix = Index(cp)
        sv = SQLView(cp)
        gs = gs_ids(cp)
        qs = [(*rng.sample(gs, 2), rng.randint(0, 59)) for _ in range(nq)]
        t_ea, t_sweep, t_chk, t_sql = [], [], [], []
        for i, (s, d, k0) in enumerate(qs):
            t = time.perf_counter(); f, par, _ = earliest_arrival(ix, s, k0, T - 1, stop_at=d); t_ea.append(time.perf_counter() - t)
            t = time.perf_counter(); fa, _, _ = earliest_arrival(ix, s, k0, T - 1); t_sweep.append(time.perf_counter() - t)
            if f[d] < INF:
                J = journey_from_parent(par, s, d)
                t = time.perf_counter(); ok = check_answer(ix, "EARLIEST", s, d, k0, T - 1, f[d], J, f); t_chk.append(time.perf_counter() - t)
                assert ok
            if i < nq_sql:
                t = time.perf_counter(); r = sv.earliest(s, d, k0, T - 1); t_sql.append(time.perf_counter() - t)
                assert r == (f[d] if f[d] < INF else None)
        ms = lambda xs, q: 1000 * pct(xs, q)
        rows.append(dict(hours=h, T=T, sats=P * S, contacts=cp.n_contacts, vnode=sv.n_vnode, vedge=sv.n_vedge,
                         mat_ms=1000 * sv.materialise_s, mem_mb=sv.mem_mb,
                         ea50=ms(t_ea, 50), ea95=ms(t_ea, 95), sw50=ms(t_sweep, 50), sw95=ms(t_sweep, 95),
                         chk50=ms(t_chk, 50), chk95=ms(t_chk, 95), sql50=ms(t_sql, 50), sql95=ms(t_sql, 95),
                         nq=nq, nq_sql=nq_sql))
        print("horizon", rows[-1], flush=True)
        del ix, sv, cp; gc.collect()
    return rows


def exp_sensitivity(quick, rng):
    """Zero-duration abstraction vs finer slots and one-hop-per-slot transmissions."""
    configs = [(60, 0), (10, 0), (10, 1)]
    regimes = ["isl_full", "isl_intra", "no_isl"]
    nq = 40 if quick else 300
    out = {}
    for reg in regimes:
        base_q = [(*rng.sample(range(66, 86), 2), rng.uniform(0, 120)) for _ in range(nq)]
        res_default = None
        for (dt, tau) in configs:
            T = int(180 * 60 / dt)
            cp = make_contact_plan(6, 11, T, reg, dt=dt)
            ix = Index(cp)
            W = int(60 * 60 / dt)
            exists, delay, hops, static_err = [], [], [], 0
            for (s, d, t0) in base_q:
                k0 = int(t0 * 60 / dt); k1 = min(k0 + W, T - 1)
                f, par, _ = earliest_arrival(ix, s, k0, k1, tau=tau)
                ok = f[d] < INF
                exists.append(ok)
                if ok:
                    delay.append((f[d] - k0) * dt / 60.0)
                    hops.append(len(journey_from_parent(par, s, d)))
                static_err += (static_reach(ix, s, d, k0, k1) != ok)
            if res_default is None:
                res_default = prev = exists
            changed = sum(a != b for a, b in zip(exists, res_default)) / nq
            changed_prev = sum(a != b for a, b in zip(exists, prev)) / nq   # vs. the previous row
            prev = exists
            out[f"{reg}|{dt}|{tau}"] = dict(reach=sum(exists) / nq, delay50=med(delay), delay95=pct(delay, 95),
                                          hops50=med(hops), hops_max=max(hops) if hops else 0,
                                          changed=changed, changed_prev=changed_prev,
                                          static_err=static_err / nq, nq=nq)
            print("sens", reg, dt, tau, out[f"{reg}|{dt}|{tau}"], flush=True)
            del ix, cp; gc.collect()
    return out


def exp_mutation(quick, rng):
    nq = 40 if quick else 200
    tot = dict(tried=0, rejected=0, valid_n=0, valid_ok=0, classes=set())
    per = {}
    for reg in ("isl_full", "isl_intra", "no_isl"):
        cp = make_contact_plan(6, 11, T_SLOTS, reg); ix = Index(cp)
        gs = gs_ids(cp)
        qs = []
        for _ in range(nq):
            s, d = rng.sample(gs, 2); k0 = rng.randint(0, 110); qs.append((s, d, k0, k0 + 60))
        for tau in (0, 1):
            r = mutation.run(ix, qs, tau, seed=rng.randint(0, 10**6))
            tot["tried"] += sum(r["tried"].values()); tot["rejected"] += sum(r["rejected"].values())
            tot["valid_n"] += r["valid_n"]; tot["valid_ok"] += r["valid_ok"]
            tot["classes"] |= set(r["tried"])
            for c, v in r["component_accepts"].items():
                tot.setdefault("component_accepts", {}); tot["component_accepts"][c] = tot["component_accepts"].get(c, 0) + v
            for c in r["tried"]:
                per.setdefault(c, [0, 0]); per[c][0] += r["tried"][c]; per[c][1] += r["rejected"][c]
    tot["classes"] = len(tot["classes"])
    tot["per_class"] = per
    print("mutation", tot, flush=True)
    return tot


def machine_info():
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    ram = float("nan")
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                ram = int(line.split()[1]) / 2**20; break
    except Exception:
        pass
    import duckdb
    return dict(cpu=cpu, ram_gb=ram, python=platform.python_version(), numpy=np.__version__,
                duckdb=duckdb.__version__)


def thou(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def write_outputs(res):
    os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "gen"), exist_ok=True)
    scal, enum, ws = res["scal"], res["enum"], res["ws"]
    for reg in ws:                                   # JSON round-trip turns int keys into strings
        ws[reg]["sem"] = {int(k): v for k, v in ws[reg]["sem"].items()}

    # Figure: (a) latency vs size, (b) expansions vs hop bound
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.45))
    x = [r["sats"] for r in scal]
    for key, qk, fmt_, lab in (("py_ms", "py_q", "o-", "TEG engine (Python)"),
                                ("sql_ms", "sql_q", "s--", "TEG view (DuckDB, rec. CTE)")):
        y = np.array([r[key] for r in scal])
        lo = y - np.array([r[qk][0] for r in scal]); hi = np.array([r[qk][1] for r in scal]) - y
        ax[0].errorbar(x, y, yerr=[lo, hi], fmt=fmt_, capsize=1.5, elinewidth=0.6, label=lab)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(v) for v in x]); ax[0].minorticks_off()
    ax[0].set_xlabel("satellites"); ax[0].set_ylabel("query time [ms], median, IQR")
    ax[0].set_ylim(0.5, 2e4)
    ax[0].legend(loc="upper left", handlelength=1.6)
    ax[0].set_title("(a) EARLIEST, 60-slot window", fontsize=7)
    H = [r["H"] for r in enum]
    ax[1].plot(H, [r["post"] for r in enum], "v-", label="enumerate + post-filter")
    ax[1].plot(H, [r["pruned"] for r in enum], "^-", label="enumerate + prune")
    ax[1].plot(H, [r["teg"] for r in enum], "o-", label="hop-bounded TEG view")
    ax[1].axhline(enum[0]["cap"], color="gray", lw=0.5, ls=":")
    ax[1].text(H[0], enum[0]["cap"] * 0.2, "cap", color="gray", fontsize=6, ha="left")
    ax[1].set_yscale("log"); ax[1].set_xlabel("hop bound $H$"); ax[1].set_ylabel("expanded states")
    ax[1].set_ylim(top=enum[0]["cap"] * 1e5)
    ax[1].set_ylim(bottom=10)
    ax[1].set_yticks([1e1, 1e3, 1e5, 1e7])
    ax[1].legend(loc="upper left", handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
    ax[1].set_title("(b) 66 sats, 30-slot window", fontsize=7)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(os.path.join(OUTDIR, "figs", "fig_scal.pdf"))
    plt.close(fig)

    # Table: time-agnostic semantics
    name = {"isl_full": "Full ISLs", "isl_intra": "Intra-plane ISLs", "no_isl": "No ISLs (S\\&F)"}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Regime & $W$ [min] & Reachable & \multicolumn{1}{c}{Static} & \multicolumn{1}{c}{Snapshot} \\",
             r" & & (truth) & error & error \\", r"\midrule"]
    for reg in ["isl_full", "isl_intra", "no_isl"]:
        for j, W in enumerate((15, 30, 60)):
            s = ws[reg]["sem"][W]
            lines.append(f"{name[reg] if j == 0 else ''} & {W} & {100*s['truth']:.1f}\\% & "
                         f"{100*s['err_static']:.1f}\\% & {100*s['err_snap']:.1f}\\% \\\\")
        if reg != "no_isl":
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    # fix column spec (5 cols)
    lines[0] = r"\begin{tabular}{@{}llccc@{}}"
    open(os.path.join(OUTDIR, "gen", "tab_semantics.tex"), "w").write("\n".join(lines) + "\n")

    # Table: witnesses and certificates
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r" & Full ISLs & Intra-plane & No ISLs \\", r"\midrule"]
    regs = ["isl_full", "isl_intra", "no_isl"]
    def row(label, fmt):
        return label + " & " + " & ".join(fmt(ws[r]) for r in regs) + r" \\"
    lines += [
        row("Reachable queries", lambda R: f"{R['reach']}/{R['nq']}"),
        row("Median model delay [min]", lambda R: f"{R['delay_med']:.0f}"),
        row("Witness $|W_C|$ (median / max)", lambda R: f"{R['L_med']:.0f} / {R['L_max']}"),
        row("Coarse witness (median)", lambda R: f"{R['coarse_med']:.0f}"),
        row("Size reduction (median)", lambda R: f"{100*R['red_med']:.1f}\\%"),
        row("Verify $J$ [$\\mu$s]", lambda R: f"{R['tj_us']:.0f}"),
        row("Verify $f$ [ms]", lambda R: f"{R['tc_ms']:.2f}"),
        r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUTDIR, "gen", "tab_witness.tex"), "w").write("\n".join(lines) + "\n")

    # Table: horizon scaling
    hor = res["hor"]
    lines = [r"\begin{tabular}{@{}rcrrcccc@{}}", r"\toprule",
             r"$T$ & View & Mat. & Mem. & \EA{} & Sweep & DuckDB & Check \\",
             r"{[h]} & nodes/edges & [ms] & [MB] & [ms] & [ms] & [ms] & [ms] \\", r"\midrule"]
    for r in hor:
        lines.append(f"{r['hours']} & {r['vnode']/1e3:.0f}k/{r['vedge']/1e6:.1f}M & {r['mat_ms']:.0f} & {r['mem_mb']:.0f} & "
                     f"{r['ea50']:.2f}/{r['ea95']:.2f} & {r['sw50']:.0f}/{r['sw95']:.0f} & {r['sql50']:.0f} & {r['chk50']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUTDIR, "gen", "tab_horizon.tex"), "w").write("\n".join(lines) + "\n")

    # Table: sensitivity to the zero-duration abstraction
    sens = res["sens"]
    cfg_name = {(60, 0): r"60\,s, $\tau{=}0$", (10, 0): r"10\,s, $\tau{=}0$", (10, 1): r"10\,s, $\tau{=}1$"}
    reg_name = {"isl_full": "Full ISLs", "isl_intra": "Intra-plane", "no_isl": "No ISLs"}
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Regime & $\Delta$, $\tau$ & Reach. & Delay & Hops & Chg. & Static \\",
             r" & & & p50 [min] & p50/max & & err. \\", r"\midrule"]
    for reg in ("isl_full", "isl_intra", "no_isl"):
        for j, cfg in enumerate(((60, 0), (10, 0), (10, 1))):
            R = sens[f"{reg}|{cfg[0]}|{cfg[1]}"]
            chg = "--" if j == 0 else f"{100*R['changed_prev']:.1f}\\%"
            lines.append(f"{reg_name[reg] if j == 0 else ''} & {cfg_name[cfg]} & {100*R['reach']:.0f}\\% & "
                         f"{R['delay50']:.1f} & {R['hops50']:.0f}/{R['hops_max']} & {chg} & "
                         f"{100*R['static_err']:.0f}\\% \\\\")
        if reg != "no_isl":
            lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUTDIR, "gen", "tab_sens.tex"), "w").write("\n".join(lines) + "\n")

    # Macros for inline numbers
    big = scal[-1]
    tot_agree = sum(r["agree"] for r in scal); tot_q = sum(r["nq"] for r in scal)
    total_reach = sum(ws[r]["reach"] for r in regs)
    m = {
        "NsatMax": big["sats"], "ContactsMax": f"{big['contacts']:,}".replace(",", "{,}"),
        "VnodeMax": f"{big['vnode']:,}".replace(",", "{,}"), "VedgeMax": f"{big['vedge']:,}".replace(",", "{,}"),
        "MatMsMax": f"{big['mat_ms']:.0f}", "PyMsMax": f"{big['py_ms']:.1f}", "SqlMsMax": f"{big['sql_ms']:.0f}",
        "PyMsMin": f"{scal[0]['py_ms']:.1f}", "SqlMsMin": f"{scal[0]['sql_ms']:.0f}",
        "AgreeQ": tot_agree, "TotQ": tot_q,
        "EnumPostHthree": thou(enum[2]['post']) if len(enum) > 2 else "--",
        "EnumPruneHmax": f"{enum[-1]['pruned']:.0f}", "TegHmax": f"{enum[-1]['teg']:.0f}",
        "EnumHmax": enum[-1]["H"],
        "TegHfour": thou(enum[3]['teg']) if len(enum) > 3 else "--",
        "CappedPostHfour": enum[3]["capped_post"] if len(enum) > 3 else "--",
        "CappedPruneHfour": enum[3]["capped_pruned"] if len(enum) > 3 else "--",
        "EnumPruneHthree": thou(enum[2]['pruned']) if len(enum) > 2 else "--",
        "TegHthree": thou(enum[2]['teg']) if len(enum) > 2 else "--", "EnumCap": f"10^{int(np.log10(enum[0]['cap']))}",
        "CappedPostHmax": enum[-1]["capped_post"], "CappedPruneHmax": enum[-1]["capped_pruned"], "EnumNq": enum[0]["nq"],
        "ReplayOk": sum(ws[r]["replay_ok"] for r in regs), "TotalReach": total_reach,
        "SimpleOk": sum(ws[r]["simple_ok"] for r in regs),
        "MinimalOk": sum(ws[r]["minimal_ok"] for r in regs), "MinimalN": sum(ws[r]["minimal_n"] for r in regs),
        "VerifiedOk": sum(ws[r]["verified"] for r in regs), "TotalWQ": sum(ws[r]["nq"] for r in regs),
        "StaticErrNoIslSixty": f"{100*ws['no_isl']['sem'][60]['err_static']:.1f}",
        "SnapErrNoIslSixty": f"{100*ws['no_isl']['sem'][60]['err_snap']:.1f}",
        "StaticErrIntraSixty": f"{100*ws['isl_intra']['sem'][60]['err_static']:.1f}",
        "SnapErrIntraSixty": f"{100*ws['isl_intra']['sem'][60]['err_snap']:.1f}",
        "StaticErrFullSixty": f"{100*ws['isl_full']['sem'][60]['err_static']:.1f}",
        "SnapErrFullSixty": f"{100*ws['isl_full']['sem'][60]['err_snap']:.1f}",
        "NqSem": ws["isl_full"]["nq"],
        "StaticNoIslMin": f"{100*min(ws['no_isl']['sem'][W]['err_static'] for W in (15, 30, 60)):.0f}",
        "StaticNoIslMax": f"{100*max(ws['no_isl']['sem'][W]['err_static'] for W in (15, 30, 60)):.0f}",
        "SnapNoIslMin": f"{100*min(ws['no_isl']['sem'][W]['err_snap'] for W in (15, 30, 60)):.0f}",
        "SnapNoIslMax": f"{100*max(ws['no_isl']['sem'][W]['err_snap'] for W in (15, 30, 60)):.0f}",
        "SnapIslMin": f"{100*min(ws[r]['sem'][W]['err_snap'] for r in ('isl_full', 'isl_intra') for W in (15, 30, 60)):.0f}",
        "SnapIslMax": f"{100*max(ws[r]['sem'][W]['err_snap'] for r in ('isl_full', 'isl_intra') for W in (15, 30, 60)):.0f}",
        "RedMin": f"{100*min(ws[r]['red_med'] for r in regs):.0f}",
        "RedMax": f"{100*max(ws[r]['red_med'] for r in regs):.0f}",
        "LMedMin": f"{min(ws[r]['L_med'] for r in regs):.0f}",
        "LMedMax": f"{max(ws[r]['L_med'] for r in regs):.0f}",
    }
    mut, mach = res["mut"], res["machine"]
    h0, h1 = hor[0], hor[-1]
    sens = res["sens"]
    chg = [sens[k]["changed"] for k in sens if not k.endswith("|60|0")]
    m.update({
        "MutTried": thou(mut["tried"]), "MutRejected": thou(mut["rejected"]), "MutClasses": mut["classes"],
        "MutValidN": thou(mut["valid_n"]), "MutValidOk": thou(mut["valid_ok"]),
        "MutForged": thou(sum(v[0] for c, v in mut["per_class"].items() if c.endswith("forged_lb"))),
        "MutForgedCompAcc": thou(sum(mut.get("component_accepts", {}).values())),
        "CPU": mach["cpu"].replace("(R)", "").replace("(TM)", "").replace("  ", " "),
        "RAMGB": f"{mach['ram_gb']:.0f}", "PyVer": mach["python"], "NpVer": mach["numpy"], "DuckVer": mach["duckdb"],
        "HorSats": h0["sats"], "HorMin": h0["hours"], "HorMax": h1["hours"], "HorNq": h0["nq"], "HorNqSql": h0["nq_sql"],
        "HorVedgeMax": f"{h1['vedge']/1e6:.1f}", "HorMatMax": f"{h1['mat_ms']:.0f}", "HorMemMax": f"{h1['mem_mb']:.0f}",
        "HorEaMin": f"{h0['ea50']:.2f}", "HorEaMax": f"{h1['ea50']:.2f}",
        "HorSweepMin": f"{h0['sw50']:.0f}", "HorSweepMax": f"{h1['sw50']:.0f}",
        "HorSqlMin": f"{h0['sql50']:.0f}", "HorSqlMax": f"{h1['sql50']:.0f}",
        "HorChkMax": f"{max(r['chk50'] for r in hor):.2f}",
        "SensChangedMax": f"{100*max(chg):.0f}",
        "SensSlotChgNoIsl": f"{100*sens['no_isl|10|0']['changed_prev']:.0f}",
        "SensTauChgNoIsl": f"{100*sens['no_isl|10|1']['changed_prev']:.1f}",
        "SensReachNoIslSixty": f"{100*sens['no_isl|60|0']['reach']:.0f}",
        "SensReachNoIslTen": f"{100*sens['no_isl|10|0']['reach']:.0f}",
        "SensTauDelayIsl": f"{max(sens['isl_full|10|1']['delay50'], sens['isl_intra|10|1']['delay50']):.1f}",
        "StaticAllMin": f"{min(100*min(sens[k]['static_err'] for k in sens if k.startswith('no_isl')), 100*min(ws['no_isl']['sem'][W]['err_static'] for W in (15, 30, 60))):.0f}",
        "StaticAllMax": f"{max(100*max(sens[k]['static_err'] for k in sens if k.startswith('no_isl')), 100*max(ws['no_isl']['sem'][W]['err_static'] for W in (15, 30, 60))):.0f}",
        "SensStaticNoIslMin": f"{100*min(sens[k]['static_err'] for k in sens if k.startswith('no_isl')):.0f}",
        "SensStaticNoIslMax": f"{100*max(sens[k]['static_err'] for k in sens if k.startswith('no_isl')):.0f}",
        "SensNq": sens["isl_full|60|0"]["nq"],
        "ScalNq": scal[0]["nq"],
    })
    with open(os.path.join(OUTDIR, "gen", "macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")


PARTS = ["scal", "enum", "ws", "hor", "sens", "mut"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=",".join(PARTS),
                    help="comma-separated subset of " + ",".join(PARTS) + " (results are merged into results.json)")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "results.json")
    res = {}
    if os.path.exists(path) and a.only != ",".join(PARTS):
        res = json.load(open(path))
    fns = dict(scal=exp_scalability, enum=exp_enumeration, ws=exp_witness_and_semantics,
               hor=exp_horizon, sens=exp_sensitivity, mut=exp_mutation)
    for part in a.only.split(","):
        rng = random.Random(f"{SEED}-{part}")          # independent, reproducible stream per part
        t = time.time()
        res[part] = fns[part](a.quick, rng)
        print(f"[{part}] {time.time() - t:.1f}s", flush=True)
    res["machine"] = machine_info()
    json.dump(res, open(path, "w"), indent=1, default=str)
    res = json.loads(json.dumps(res, default=str))  # normalise keys (e.g. int dict keys -> str)
    if all(p in res for p in PARTS):
        write_outputs(res)
        print("outputs written")


if __name__ == "__main__":
    main()
