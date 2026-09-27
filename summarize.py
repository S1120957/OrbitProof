"""Summarise LLM-study logs (harness.py --out) into a results table (LaTeX rows + macros) and an analysis report.

  python3 summarize.py logs/*.jsonl                 # report only -> llm_summary.md
  python3 summarize.py logs/*.jsonl --write-tables   # also writes out/gen/tab_llm.tex, llm_macros.tex
  python3 summarize.py --reset-tables                # restore the TBD results table and remove llm_macros.tex

--write-tables refuses to run if any record comes from a mock provider (mock-gold, mock-naive):
those are pipeline tests, not LLM results, and must never be reported as results.

Reports, per system x LLM: accuracy (Wilson 95% CI), coverage, accuracy of released answers,
wrong-but-released, rejection/error rate, exact IR match; exact McNemar tests of OrbitProof vs each
baseline on paired instances; per-template accuracy; and an error taxonomy (timeouts, SQL errors,
time-agnostic answers, other wrong answers; invalid IRs and the IR fields that were misread).
"""
from __future__ import annotations
import argparse, collections, glob, json, math, os
from harness import load_env, naive_sql, run_sql, norm_answer

SYSTEMS = [("sql", r"Text-to-SQL, base tables"), ("sql_hint", r"\quad + semantics hint"),
           ("view_sql", r"Text-to-SQL, TEG view"), ("ir", r"OrbitProof (IR + checker)")]
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(os.environ.get("ORBITPROOF_OUT", os.path.join(HERE, "out")), "gen")


SYS_ROWS = [("sql", "Text-to-SQL (base)"), ("sql_hint", "\\quad + hint"),
            ("view_sql", "Text-to-SQL (view)"), ("ir", "OrbitProof")]
REF_ROWS = [r"Time-agnostic ref. & -- & \RNaiveAcc\% & \RNaiveAccCov\% & \RNaiveAccAccRel\% & \RNaiveAccWrongRel\% & \RNaiveAccRej\% & -- \\",
            r"Gold IR (upper bound) & -- & \RGoldIRAcc\% & \RGoldIRAccCov\% & \RGoldIRAccAccRel\% & \RGoldIRAccWrongRel\% & \RGoldIRAccRej\% & \RGoldIRAccIRx\% \\"]


def llm_table(rows, labels):
    """Results table: one row per (system, model); rows[(system, label)] -> metrics or None for TBD."""
    t = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"System & LLM & Acc. & Cov. & Acc.$|$rel. & Wrong & Err. & IR ex. \\", r"\midrule"]
    for k, (sysname, disp) in enumerate(SYS_ROWS):
        for j, lab in enumerate(labels):
            R = rows.get((sysname, lab))
            name = disp if j == 0 else ""
            irx = ("\\tbd" if R is None else f"{R['irx']:.1f}\\%") if sysname == "ir" else "--"
            if R is None:
                cells = ["\\tbd"] * 5
            else:
                cells = [f"{R[m]:.1f}\\%" for m in ("acc", "cov", "accrel", "wrong", "rej")]
            t.append(f"{name} & Model {lab} & " + " & ".join(cells) + f" & {irx} \\\\")
        if k < len(SYS_ROWS) - 1:
            t.append(r"\addlinespace[1.5pt]")
    t += [r"\midrule"] + REF_ROWS + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(t) + "\n"


TBD_TABLE = llm_table({}, ["A", "B"])


def llm_macros(rows, models, holm):
    """LaTeX macros for the reported metrics (per system and model, model ids, Holm maxima)."""
    m = {}
    for (sysname, lab), R in rows.items():
        key = {"sql": "Sql", "sql_hint": "Hint", "view_sql": "View", "ir": "Ours"}[sysname] + \
              "".join(ch for ch in lab if ch.isalpha())
        for name, field in (("Acc", "acc"), ("Cov", "cov"), ("AccRel", "accrel"), ("Wrong", "wrong"),
                            ("Err", "rej"), ("IRx", "irx")):
            m[f"LLM{key}{name}"] = f"{R[field]:.1f}"
    for lab, model in models.items():
        m[f"LLMModel{''.join(ch for ch in str(lab) if ch.isalpha())}"] = f"\\texttt{{\\detokenize{{{model}}}}}"
    for lab, p in holm.items():
        m[f"LLMHolmMax{''.join(ch for ch in str(lab) if ch.isalpha())}"] = f"{p:.1g}".replace("e-0", "e-")
    return "".join(f"\\newcommand{{\\R{k}}}{{{v}}}\n" for k, v in m.items())


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def naive_answers(bench_path):
    items = [json.loads(l) for l in open(bench_path)]
    envs, out = {}, {}
    for it in items:
        cp, ix, sv = load_env(it["regime"], envs)
        v, err = run_sql(sv.con, naive_sql(cp.names, it))
        out[it["id"]] = None if err else norm_answer(it["gold_ir"]["task"], v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--bench", default=os.path.join(HERE, "bench.jsonl"))
    ap.add_argument("--write-tables", action="store_true")
    ap.add_argument("--reset-tables", action="store_true", help="restore the TBD results table and exit")
    a = ap.parse_args()
    if a.reset_tables:
        os.makedirs(GEN, exist_ok=True)
        open(os.path.join(GEN, "tab_llm.tex"), "w").write(TBD_TABLE)
        mp = os.path.join(GEN, "llm_macros.tex")
        if os.path.exists(mp):
            os.remove(mp)
        print("restored TBD results table (out/gen/tab_llm.tex); removed llm_macros.tex")
        return
    paths = sorted({p for g in a.logs for p in glob.glob(g) if not p.endswith(".raw.jsonl")})
    recs = [json.loads(l) for p in paths for l in open(p)]
    if not recs:
        raise SystemExit("no records")
    providers = {r.get("provider") for r in recs}
    for p in paths:                                   # older logs: provider only in the raw cache
        rp = p + ".raw.jsonl"
        if os.path.exists(rp):
            providers |= {json.loads(l).get("provider") for l in open(rp)}
    providers.discard(None)
    mock = sorted(pv for pv in providers if str(pv).startswith("mock"))
    naive = naive_answers(a.bench)
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["system"], r.get("label") or r.get("model") or "?")].append(r)
    labels = sorted({k[1] for k in groups})

    rows, lines = {}, ["# LLM study summary", "", f"Logs: {len(paths)} files, {len(recs)} records; prompt versions: "
                       + ", ".join(sorted({r.get('prompt_sha', '?') for r in recs}))
                       + "; providers: " + (", ".join(sorted(map(str, providers))) or "unknown"), ""]
    if mock:
        lines += [f"> **PIPELINE TEST ONLY.** Records from mock providers {mock}: `mock-gold` returns the gold "
                  "IR/SQL by construction and `mock-naive` returns a fixed time-agnostic query for SQL systems "
                  "(and the gold IR for `ir`). These numbers are not LLM results and must not be reported.", ""]
    lines += ["| System | LLM | n | Acc. [95% CI] | Cov. | Acc.\\|rel. | Wrong-rel. | Rej. | IR exact |",
              "|---|---|---|---|---|---|---|---|---|"]
    for sysname, _ in SYSTEMS:
        for lab in labels:
            g = groups.get((sysname, lab))
            if not g:
                continue
            n = len(g); k = sum(r["correct"] for r in g); rel = sum(r["released"] for r in g)
            wr = sum(r["released"] and not r["correct"] for r in g)
            irx = sum(r.get("ir_exact", False) for r in g)
            lo, hi = wilson(k, n)
            rows[(sysname, lab)] = dict(n=n, acc=100 * k / n, cov=100 * rel / n,
                                        accrel=100 * k / max(rel, 1), wrong=100 * wr / n,
                                        rej=100 * (n - rel) / n, irx=100 * irx / n, lo=lo, hi=hi)
            R = rows[(sysname, lab)]
            lines.append(f"| {sysname} | {lab} | {n} | {R['acc']:.1f} [{lo:.1f}, {hi:.1f}] | {R['cov']:.1f} | "
                         f"{R['accrel']:.1f} | {R['wrong']:.1f} | {R['rej']:.1f} | "
                         f"{R['irx']:.1f} |" if sysname == "ir" else
                         f"| {sysname} | {lab} | {n} | {R['acc']:.1f} [{lo:.1f}, {hi:.1f}] | {R['cov']:.1f} | "
                         f"{R['accrel']:.1f} | {R['wrong']:.1f} | {R['rej']:.1f} | -- |")

    # internal-consistency invariants (hold by construction of the pipeline)
    viol = []
    for (sysname, lab), R in rows.items():
        if abs(R["acc"] + R["wrong"] + R["rej"] - 100.0) > 0.05:
            viol.append(f"{sysname}/{lab}: acc + wrong + err != 100")
        if sysname == "ir" and R["acc"] + 1e-9 < R["irx"]:
            viol.append(f"ir/{lab}: accuracy below exact-IR rate (an exact IR always yields the gold answer)")
    lines += ["", "## Consistency checks", "",
              "- " + ("; ".join(viol) if viol else "all invariants hold (acc + wrong + err = 100; OrbitProof acc >= IR-exact)")]

    # paired tests: OrbitProof vs each baseline, same LLM, same instances
    lines += ["", "## Paired comparison (exact McNemar, OrbitProof vs baseline, same LLM; Holm-adjusted over all comparisons)", "",
              "| LLM | Baseline | both right | only OrbitProof | only baseline | both wrong | p | p (Holm) |",
              "|---|---|---|---|---|---|---|---|"]
    tests = []
    for lab in labels:
        ours = {r["id"]: r["correct"] for r in groups.get(("ir", lab), [])}
        for sysname, _ in SYSTEMS[:-1]:
            base = {r["id"]: r["correct"] for r in groups.get((sysname, lab), [])}
            ids = ours.keys() & base.keys()
            if not ids:
                continue
            both = sum(ours[i] and base[i] for i in ids); b = sum(ours[i] and not base[i] for i in ids)
            c = sum(base[i] and not ours[i] for i in ids); nn = sum(not ours[i] and not base[i] for i in ids)
            tests.append([lab, sysname, both, b, c, nn, mcnemar_exact(b, c)])
    order = sorted(range(len(tests)), key=lambda i: tests[i][6])
    running = 0.0
    for rank, i in enumerate(order):                   # Holm step-down, monotone
        running = max(running, min(1.0, (len(tests) - rank) * tests[i][6]))
        tests[i].append(running)
    for lab, sysname, both, b, c, nn, p, ph in tests:
        lines.append(f"| {lab} | {sysname} | {both} | {b} | {c} | {nn} | {p:.2g} | {ph:.2g} |")

    # per-regime accuracy
    regs = sorted({r["regime"] for r in recs})
    lines += ["", "## Accuracy per regime", "", "| System | LLM | " + " | ".join(regs) + " |",
              "|---|---|" + "---|" * len(regs)]
    for (sysname, lab), g in sorted(groups.items()):
        cells = []
        for rg in regs:
            gr = [r for r in g if r["regime"] == rg]
            cells.append(f"{100*sum(r['correct'] for r in gr)/max(len(gr),1):.1f}" if gr else "--")
        lines.append(f"| {sysname} | {lab} | " + " | ".join(cells) + " |")

    # per-template accuracy
    tpls = sorted({r["template"] for r in recs})
    lines += ["", "## Accuracy per template", "", "| System | LLM | " + " | ".join(tpls) + " |",
              "|---|---|" + "---|" * len(tpls)]
    for (sysname, lab), g in sorted(groups.items()):
        cells = []
        for t in tpls:
            gt = [r for r in g if r["template"] == t]
            cells.append(f"{100*sum(r['correct'] for r in gt)/max(len(gt),1):.0f}" if gt else "--")
        lines.append(f"| {sysname} | {lab} | " + " | ".join(cells) + " |")

    # error taxonomy
    lines += ["", "## Error taxonomy", ""]
    for (sysname, lab), g in sorted(groups.items()):
        tax = collections.Counter()
        for r in g:
            if r["correct"]:
                continue
            if sysname == "ir":
                if not r["released"]:
                    tax["invalid/ungrounded IR or rejected"] += 1
                else:
                    tax["certified but wrong (IR misread)"] += 1
            else:
                err = r.get("error") or ""
                if err == "timeout":
                    tax["timeout"] += 1
                elif err:
                    tax["SQL error"] += 1
                elif r["pred"] == naive.get(r["id"]) and naive.get(r["id"]) != r["gold"]:
                    tax["time-agnostic answer"] += 1
                else:
                    tax["other wrong answer"] += 1
        lines.append(f"- **{sysname} / {lab}**: " + (", ".join(f"{k} {v}" for k, v in tax.most_common()) or "no errors"))

    # IR field-level misreads
    bench = {json.loads(l)["id"]: json.loads(l) for l in open(a.bench)}
    field_err = collections.defaultdict(collections.Counter)
    for (sysname, lab), g in groups.items():
        if sysname != "ir":
            continue
        for r in g:
            if "ir" in r and not r.get("ir_exact"):
                gold = bench[r["id"]]["gold_ir"]
                for fld in gold:
                    if r["ir"].get(fld) != gold.get(fld):
                        field_err[lab][fld] += 1
    if field_err:
        lines += ["", "## IR fields misread (released or not)", ""]
        for lab, cnt in field_err.items():
            lines.append(f"- **{lab}**: " + ", ".join(f"{k} {v}" for k, v in cnt.most_common()))

    os.makedirs(os.path.dirname(GEN), exist_ok=True)
    open(os.path.join(os.path.dirname(GEN), "llm_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if a.write_tables and mock:
        raise SystemExit(f"REFUSED: logs contain mock providers {mock}. Mock runs test the pipeline and are "
                         f"not LLM results; the results table was not written. Run with real providers, or "
                         f"'python3 summarize.py --reset-tables' to restore the TBD table.")
    if a.write_tables and not providers:
        raise SystemExit("REFUSED: provider of the logs is unknown; re-score them with the current harness.")
    if a.write_tables:
        os.makedirs(GEN, exist_ok=True)
        open(os.path.join(GEN, "tab_llm.tex"), "w").write(llm_table(rows, labels))
        models = {}
        for r in recs:
            models.setdefault(r.get("label") or r.get("model"), r.get("model"))
        holm = {}
        for lab, sysname, both, b, c, nn, p, ph in tests:
            holm[lab] = max(holm.get(lab, 0.0), ph)
        open(os.path.join(GEN, "llm_macros.tex"), "w").write(llm_macros(rows, models, holm))
        print("wrote out/gen/tab_llm.tex and llm_macros.tex")


if __name__ == "__main__":
    main()
