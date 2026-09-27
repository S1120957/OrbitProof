"""Generates the NL question benchmark (JSONL) for the LLM study.

Each item: question text, gold IR, gold answer (from the certified engine),
regime, template id. Gold answers are produced by agent.execute() and are
cross-checked against the SQL compiled from the gold IR (DuckDB).
Usage: python3 make_benchmark.py --per-template 50 --out bench.jsonl
"""
from __future__ import annotations
import argparse, json, random
from constellation import make_contact_plan
from teg import Index
from agent import JourneyIR, execute, slot_to_hhmm, compile_sql
from sqlview import SQLView

DISPLAY = {"NewYork": "New York", "SaoPaulo": "São Paulo", "CapeTown": "Cape Town",
           "LosAngeles": "Los Angeles"}
T = 180

TEMPLATES = {
    "T1_exists": [
        "Can a bundle sent from {S} at {t0} UTC reach {D} by {t1} UTC?",
        "If {S} hands over a message at {t0} UTC, will {D} have it no later than {t1} UTC?",
        "Is delivery {S} -> {D} possible between {t0} and {t1} UTC with the current contact plan?",
    ],
    "T2_earliest": [
        "What is the earliest time a message leaving {S} at {t0} UTC can be delivered to {D}?",
        "{S} has data ready at {t0} UTC. When is the soonest {D} can receive it?",
        "Earliest arrival at {D} for traffic injected at {S} at {t0} UTC?",
    ],
    "T3_avoid": [
        "Satellite {X} is out of service. What is the earliest a bundle from {S} sent at {t0} UTC can reach {D}?",
        "Excluding {X}, when can {D} earliest receive data that {S} sends at {t0} UTC?",
        "With {X} unavailable, earliest delivery from {S} (ready {t0} UTC) to {D}?",
    ],
    "T4_via": [
        "A bundle must pass through the {G} ground station. Leaving {S} at {t0} UTC, when is the earliest it can reach {D}?",
        "Route {S} -> {D} via {G}, departing {t0} UTC: earliest delivery time?",
        "If traffic from {S} at {t0} UTC has to be relayed by {G}, when can {D} get it at the earliest?",
    ],
    "T5_hops": [
        "Using at most {H} transmissions, can {S} get a message sent at {t0} UTC to {D} by {t1} UTC?",
        "Is there a delivery from {S} (at {t0} UTC) to {D} by {t1} UTC with no more than {H} hops?",
        "With a {H}-hop limit, can data leave {S} at {t0} UTC and arrive at {D} before {t1} UTC?",
    ],
}


def disp(n):
    return DISPLAY.get(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=50)
    ap.add_argument("--regimes", default="isl_intra,no_isl")
    ap.add_argument("--out", default="bench.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--check-sql", action="store_true", help="cross-check gold answers via compiled SQL")
    ap.add_argument("--all-paraphrases", action="store_true",
                    help="emit every paraphrase of each semantic instance (3x items) instead of one")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    items = []
    for reg in a.regimes.split(","):
        cp = make_contact_plan(6, 11, T, reg)
        ix = Index(cp)
        names = cp.names
        gs = [n for n, k in zip(names, cp.kinds) if k == "GS"]
        sv = SQLView(cp) if a.check_sql else None
        mism = 0
        for tid, pars in TEMPLATES.items():
            n_ok = 0
            tries = 0
            while n_ok < a.per_template and tries < 50 * a.per_template:
                tries += 1
                S, D, G = rng.sample(gs, 3)
                k0 = rng.randint(0, T - 70)
                W = rng.choice([15, 30, 45, 60])
                k1 = k0 + W
                if tid == "T1_exists":
                    ir = JourneyIR("EXISTS", S, D, slot_to_hhmm(k0), slot_to_hhmm(k1))
                elif tid == "T2_earliest":
                    ir = JourneyIR("EARLIEST", S, D, slot_to_hhmm(k0))
                elif tid == "T3_avoid":
                    base = execute(ix, names, JourneyIR("EARLIEST", S, D, slot_to_hhmm(k0)))
                    sats = [names[b] for (_, b, _) in (base.journey or [])[:-1] if names[b].startswith("S")]
                    if not sats:
                        continue
                    ir = JourneyIR("EARLIEST", S, D, slot_to_hhmm(k0), avoid=[rng.choice(sats)])
                    if execute(ix, names, ir).answer == base.answer:
                        continue          # the constraint must change the earliest arrival (or reachability)
                elif tid == "T4_via":
                    ir = JourneyIR("EARLIEST", S, D, slot_to_hhmm(k0), via=G)
                else:
                    ir = JourneyIR("EXISTS", S, D, slot_to_hhmm(k0), slot_to_hhmm(k1), max_hops=rng.choice([2, 3, 4]))
                res = execute(ix, names, ir)
                assert res.certified, "engine produced an uncertified answer"
                if ir.task == "EXISTS" and tries % 40 != 0 and res.answer != (n_ok % 2 == 0):
                    continue              # balance yes/no answers where the regime allows it
                if sv is not None:
                    got = sv.con.execute(compile_sql(names, ir, T)).fetchone()[0]
                    mism += (got != res.answer)
                pick = rng.choice(list(enumerate(pars)))       # always drawn, so both modes share instances
                chosen = list(enumerate(pars)) if a.all_paraphrases else [pick]
                for pi, par in chosen:
                    q = par.format(S=disp(S), D=disp(D), G=disp(G), t0=slot_to_hhmm(k0),
                                   t1=slot_to_hhmm(k1), X=(ir.avoid or [""])[0], H=ir.max_hops)
                    items.append(dict(id=f"{reg}-{tid}-{n_ok:03d}-p{pi}", instance=f"{reg}-{tid}-{n_ok:03d}",
                                      paraphrase=pi, regime=reg, template=tid, question=q,
                                      gold_ir=ir.canonical(), gold_answer=res.answer,
                                      gold_answer_hhmm=(slot_to_hhmm(res.answer) if isinstance(res.answer, int)
                                                        and not isinstance(res.answer, bool) else None)))
                n_ok += 1
        if sv is not None:
            print(f"[{reg}] compiled-SQL vs engine mismatches: {mism}")
    with open(a.out, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items ({len({i['instance'] for i in items})} semantic instances) to {a.out}")


if __name__ == "__main__":
    main()
