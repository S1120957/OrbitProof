"""Fault-injection simulation of LLM errors (no LLM is called).

Part A - intent errors in OrbitProof. For every benchmark instance, each applicable
error class perturbs the GOLD IR in one controlled way; the perturbed IR is sent through
the real pipeline (parse/ground -> compile/execute -> answer-level checker) and the outcome
is classified as benign (released, equals gold), rejected (not released) or
certified-but-wrong (released with an accepted certificate, differs from gold).

Part B - semantic ablations. For every instance, the correct semantics is evaluated with
exactly one plausible mistake (time order ignored, snapshot only, strict order, constraint
dropped, no recursion, departure ignored) by the reference engine in Python; accuracy vs
gold is reported. These are not executed SQL queries.

Part C - expected end-to-end behaviour under an explicit, stated error model: a fraction e
of IRs is erroneous and the error class is uniform over the applicable classes. This is a
projection under assumptions, not a measurement of any LLM.

Writes ../paper/gen/fi_macros.tex, tab_fi_ir.tex, tab_fi_sql.tex and fault_injection.json.
"""
from __future__ import annotations
import json, os, collections, statistics as st
from constellation import make_contact_plan
from teg import (Index, INF, earliest_arrival, earliest_arrival_hops, static_reach, snapshot_reach)
from agent import JourneyIR, IRError, parse_ir, execute, hhmm_to_slot, slot_to_hhmm

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "..", "paper", "gen")
T = 180
TEMPLATES = ["T1_exists", "T2_earliest", "T3_avoid", "T4_via", "T5_hops"]


# --------------------------------------------------------------------------- Part A
def ir_errors(g, names, gs):
    """Yield (class, raw IR text) for every class applicable to gold IR g."""
    def dump(d):
        return json.dumps(d)
    out = []
    if g["task"] == "EXISTS":
        k1 = hhmm_to_slot(g["arrive_by"])
        for delta, tag in ((-1, "deadline"), ):
            d = dict(g); d["arrive_by"] = slot_to_hhmm(max(0, k1 + delta)); out.append(("deadline", dump(d)))
    d = dict(g)
    d["depart_after"] = slot_to_hhmm(min(T - 1, hhmm_to_slot(g["depart_after"]) + 60))
    if g["arrive_by"]:
        d["arrive_by"] = slot_to_hhmm(min(T - 1, hhmm_to_slot(g["arrive_by"]) + 60))
    out.append(("timezone", dump(d)))
    if g["avoid"]:
        d = dict(g); d["avoid"] = []; out.append(("avoid_dropped", dump(d)))
    if g["via"]:
        d = dict(g); d["via"] = "Europe"; out.append(("via_region", dump(d)))
        d = dict(g); d["via"] = None; out.append(("via_dropped", dump(d)))
    if g["max_hops"]:
        d = dict(g); d["max_hops"] = g["max_hops"] - 1 if g["max_hops"] > 1 else 2; out.append(("hops_off_by_one", dump(d)))
    alt = next(n for n in gs if n not in (g["src"], g["dst"], g["via"]))
    d = dict(g); d["dst"] = alt; out.append(("wrong_station", dump(d)))
    d = dict(g)
    if g["task"] == "EXISTS":
        d["task"] = "EARLIEST"; d["arrive_by"] = None
    else:
        d["task"] = "EXISTS"; d["arrive_by"] = None
    out.append(("task_confusion", dump(d)))
    out.append(("malformed_json", dump(g)[:-5]))
    return out


def outcome(raw, it, ix, names):
    try:
        ir = parse_ir(raw, names)
        res = execute(ix, names, ir)
    except (IRError, KeyError, ValueError, TypeError):
        return "rejected"
    if not res.certified:
        return "rejected"
    return "benign" if res.answer == it["gold_answer"] else "cert_wrong"


# --------------------------------------------------------------------------- Part B
def strict_earliest(ix, s, k0, k1, avoid=frozenset(), H=None):
    """Earliest contact slot k_m of a journey with STRICTLY increasing slots (no multi-hop per slot)."""
    best = {(s, 0): k0 - 1}          # state -> slot of last contact (k0-1: nothing used yet)
    frontier = {(s, 0)}
    arrive = {}
    for k in range(k0, min(k1, ix.T - 1) + 1):
        new = set()
        for (a, h) in list(best):
            if best[(a, h)] < k and (H is None or h < H):
                for b in ix.adj[k].get(a, ()):
                    if b in avoid:
                        continue
                    key = (b, h + 1) if H is not None else (b, 0)   # hop count only matters with a bound
                    if key not in best and key not in new:
                        new.add(key)
                        arrive.setdefault(b, k)
        for st_ in new:
            best[st_] = k
    return arrive


def union_reach(ix, s, k0, k1, avoid=frozenset(), H=None):
    """Reachability and 'arrival' when time order is ignored (union of contacts in the window)."""
    cp = ix.cp
    lo, hi = ix.slot_start[k0], ix.slot_start[min(k1, cp.T - 1) + 1]
    adj = collections.defaultdict(list)
    for a, b, k in zip(cp.u[lo:hi].tolist(), cp.v[lo:hi].tolist(), cp.k[lo:hi].tolist()):
        if a not in avoid and b not in avoid:
            adj[a].append((b, k))
    depth = {s: 0}; order = [s]; first_in = {}
    for a in order:
        if H is not None and depth[a] >= H:
            continue
        for b, k in adj[a]:
            first_in[b] = min(first_in.get(b, INF), k)
            if b not in depth:
                depth[b] = depth[a] + 1; order.append(b)
    return depth, first_in


def variant_answer(var, it, ix, idx):
    g = it["gold_ir"]
    s, d = idx[g["src"]], idx[g["dst"]]
    k0 = hhmm_to_slot(g["depart_after"])
    k1 = hhmm_to_slot(g["arrive_by"]) if g["task"] == "EXISTS" else T - 1
    avoid = frozenset(idx[a] for a in g["avoid"])
    via = idx[g["via"]] if g["via"] else None
    H = g["max_hops"]
    exists = g["task"] == "EXISTS"
    if var == "no_departure":
        k0 = 0
    if var == "constraint_dropped":
        avoid, via, H = frozenset(), None, None
    if var in ("correct", "no_departure", "constraint_dropped"):
        ir = JourneyIR(g["task"], g["src"], g["dst"], slot_to_hhmm(k0), g["arrive_by"],
                       list(g["avoid"]) if avoid else [], g["via"] if via is not None else None, H)
        return execute(ix, ix.names, ir).answer
    if var == "union":
        if via is not None:
            dep1, _ = union_reach(ix, s, k0, k1, avoid)
            if via not in dep1:
                return False if exists else None
            dep2, fin = union_reach(ix, via, k0, k1, avoid)
            ok = d in dep2
        else:
            dep2, fin = union_reach(ix, s, k0, k1, avoid, H)
            ok = d in dep2
        return ok if exists else (fin.get(d) if ok else None)
    if var == "snapshot":
        f, _, _ = (earliest_arrival_hops(ix, s, k0, k0, H, avoid) if H else earliest_arrival(ix, s, k0, k0, avoid))
        reach_d = (min(f[d]) if H else f[d]) < INF
        if via is not None:
            fv, _, _ = earliest_arrival(ix, s, k0, k0, avoid)
            f2, _, _ = earliest_arrival(ix, via, k0, k0, avoid)
            reach_d = fv[via] < INF and f2[d] < INF
        return reach_d if exists else (k0 if reach_d else None)
    if var == "strict_order":
        if via is not None:
            a1 = strict_earliest(ix, s, k0, k1, avoid)
            if via not in a1:
                return False if exists else None
            a2 = strict_earliest(ix, via, a1[via] + 1, k1, avoid)
            kd = a2.get(d)
        else:
            kd = strict_earliest(ix, s, k0, k1, avoid, H).get(d)
        return (kd is not None) if exists else kd
    if var == "no_recursion":
        ks = [k for k in range(k0, min(k1, T - 1) + 1) if d in ix.adj[k].get(s, ())]
        return bool(ks) if exists else (ks[0] if ks else None)
    raise ValueError(var)


VARIANTS = [("correct", "Correct query"), ("union", "Time order ignored (union)"),
            ("snapshot", "Snapshot at departure"), ("strict_order", "Strict order (1 hop/slot)"),
            ("constraint_dropped", "Constraint dropped"), ("no_recursion", "No recursion (1 contact)"),
            ("no_departure", "Departure time ignored")]
IR_CLASSES = [("deadline", "Deadline off by 1 min"), ("timezone", "Times shifted 60 min"),
              ("avoid_dropped", "Avoid dropped"), ("via_region", "Via as region"),
              ("via_dropped", "Via dropped"), ("hops_off_by_one", "Hop bound off by one"),
              ("wrong_station", "Wrong station"), ("task_confusion", "Task confused"),
              ("malformed_json", "Malformed JSON")]


def main():
    items = [json.loads(l) for l in open(os.path.join(HERE, "bench.jsonl"))]
    envs = {}
    A = collections.defaultdict(collections.Counter)            # class -> outcome counts
    per_item_class = []                                         # for the mixture model
    B = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        reg = it["regime"]
        if reg not in envs:
            cp = make_contact_plan(6, 11, T, reg); ix = Index(cp); ix.names = cp.names
            envs[reg] = (cp, ix, {n: i for i, n in enumerate(cp.names)},
                         [n for n, k in zip(cp.names, cp.kinds) if k == "GS"])
        cp, ix, idx, gs = envs[reg]
        # Part A
        outs = []
        for cls, raw in ir_errors(it["gold_ir"], cp.names, gs):
            o = outcome(raw, it, ix, cp.names)
            A[cls][o] += 1; outs.append(o)
        per_item_class.append(outs)
        # Part B
        for var, _ in VARIANTS:
            ans = variant_answer(var, it, ix, idx)
            B[var][it["template"]].append(ans == it["gold_answer"])

    # sanity: correct variant must reproduce every gold answer
    corr = [x for t in TEMPLATES for x in B["correct"][t]]
    assert all(corr), "correct variant disagrees with gold"

    # Part C: expected behaviour under an explicit error model
    mix = {}
    for e in (0.05, 0.10, 0.20, 0.30):
        acc = cov = cw = 0.0
        for outs in per_item_class:
            p = e / len(outs)
            acc += (1 - e) + p * outs.count("benign")
            cov += 1 - p * outs.count("rejected")
            cw += p * outs.count("cert_wrong")
        n = len(per_item_class)
        mix[e] = dict(acc=100 * acc / n, cov=100 * cov / n, cw=100 * cw / n)

    tot = collections.Counter()
    for c in A.values():
        tot.update(c)
    ntot = sum(tot.values())
    res = dict(A={k: dict(v) for k, v in A.items()},
               B={v: {t: sum(x) / len(x) for t, x in B[v].items()} for v, _ in VARIANTS},
               B_all={v: sum(sum(x) for x in B[v].values()) / len(items) for v, _ in VARIANTS},
               mix={str(k): v for k, v in mix.items()}, tot=dict(tot), n_items=len(items))
    json.dump(res, open(os.path.join(HERE, "fault_injection.json"), "w"), indent=1)

    # ---- LaTeX outputs
    os.makedirs(GEN, exist_ok=True)
    L = [r"\begin{tabular}{@{}lrccc@{}}", r"\toprule",
         r"Injected intent error & $n$ & Benign & Rejected & Cert.-wrong \\", r"\midrule"]
    for cls, disp in IR_CLASSES:
        c = A[cls]; n = sum(c.values())
        L.append(f"{disp} & {n} & {100*c['benign']/n:.0f}\\% & {100*c['rejected']/n:.0f}\\% & {100*c['cert_wrong']/n:.0f}\\% \\\\")
    L += [r"\midrule", f"All injected errors & {ntot} & {100*tot['benign']/ntot:.1f}\\% & {100*tot['rejected']/ntot:.1f}\\% & {100*tot['cert_wrong']/ntot:.1f}\\% \\\\",
          r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_fi_ir.tex"), "w").write("\n".join(L) + "\n")

    short = {"T1_exists": "\\EX", "T2_earliest": "\\EA", "T3_avoid": "\\AV", "T4_via": "\\VIA", "T5_hops": "\\HOPS"}
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         "Query error & " + " & ".join(short[t] for t in TEMPLATES) + r" & All \\", r"\midrule"]
    for var, disp in VARIANTS:
        cells = [f"{100*res['B'][var][t]:.0f}" for t in TEMPLATES]
        L.append(f"{disp} & " + " & ".join(cells) + f" & {100*res['B_all'][var]:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_fi_sql.tex"), "w").write("\n".join(L) + "\n")

    wrong_vars = [v for v, _ in VARIANTS if v != "correct"]
    m = {"FiN": ntot, "FiClasses": len(IR_CLASSES),
         "FiBenign": f"{100*tot['benign']/ntot:.1f}", "FiRejected": f"{100*tot['rejected']/ntot:.1f}",
         "FiCertWrong": f"{100*tot['cert_wrong']/ntot:.1f}",
         "FiSqlMin": f"{100*min(res['B_all'][v] for v in wrong_vars):.0f}",
         "FiSqlMax": f"{100*max(res['B_all'][v] for v in wrong_vars):.0f}",
         "FiSqlVariants": len(wrong_vars)}
    for e, tag in ((0.05, "Five"), (0.10, "Ten"), (0.20, "Twenty"), (0.30, "Thirty")):
        m[f"FiMix{tag}Acc"] = f"{mix[e]['acc']:.1f}"; m[f"FiMix{tag}Cov"] = f"{mix[e]['cov']:.1f}"
        m[f"FiMix{tag}Cw"] = f"{mix[e]['cw']:.1f}"
    with open(os.path.join(GEN, "fi_macros.tex"), "w") as fh:
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")
    print(json.dumps({k: res[k] for k in ("tot", "B_all", "mix")}, indent=1))
    for cls, _ in IR_CLASSES:
        print(cls, dict(A[cls]))


if __name__ == "__main__":
    main()
