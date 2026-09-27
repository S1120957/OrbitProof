"""LLM study harness.

Systems
  sql       : LLM writes one SQL query over the base tables (node, slot, contact)
  sql_hint  : same, plus an explicit description of time-respecting journeys
  view_sql  : LLM writes one SQL query over the already-materialised TEG view (vnode, vedge):
              isolates whether the IR + compiler matter once temporal semantics are in the graph
  ir        : LLM emits the Journey IR; deterministic compiler + certifying checker (ours)

Metrics (per system): accuracy over all questions, coverage (answers released), accuracy
conditional on release, wrong-but-released (for ours: certified-but-wrong), rejection/error rate.

Providers
  anthropic   : Anthropic Messages API   (env ANTHROPIC_API_KEY, --model required)
  openai      : any OpenAI-compatible chat endpoint (env OPENAI_API_KEY, --base-url, --model)
  mock-gold   : returns the gold IR / gold-compiled SQL   (pipeline sanity check only)
  mock-naive  : returns time-agnostic SQL                 (exercises the scorer only)

Usage
  python3 make_benchmark.py --per-template 50 --out bench.jsonl
  python3 harness.py --bench bench.jsonl --system ir  --provider anthropic --model <model-id>
  python3 harness.py --bench bench.jsonl --system sql --provider openai --base-url http://localhost:8000/v1 --model <id>
"""
from __future__ import annotations
import argparse, json, os, threading, time, collections, hashlib, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from constellation import make_contact_plan
from teg import Index
from agent import (JourneyIR, IRError, parse_ir, execute, compile_sql, hhmm_to_slot, IR_SCHEMA)
from sqlview import SQLView
from make_benchmark import DISPLAY, T

SQL_TIMEOUT_S = 20

SCHEMA_TXT = """Tables (DuckDB):
  node(nid INTEGER, name VARCHAR, kind VARCHAR)   -- kind in ('SAT','GS')
  slot(k INTEGER, nk INTEGER)                      -- minute k after 00:00 UTC; nk = k+1 (NULL for last)
  contact(u INTEGER, v INTEGER, k INTEGER)         -- node u can transmit to node v during minute k
Times are minutes after 00:00 UTC (k = 60*hh + mm); the plan covers k = 0..179."""

HINT_TXT = """Delivery semantics: a message may cross several contacts within the same minute,
and may be stored at any node and forwarded later. A valid delivery is a sequence of contacts
(u0,u1,k1),(u1,u2,k2),... with depart_minute <= k1 <= k2 <= ... ; contacts must be used in
non-decreasing time order. It is NOT enough that a path exists if contacts are taken out of order."""

SQL_TASK_TXT = """Write ONE DuckDB SQL query that answers the question. It must return a single row with a
single column named answer: a BOOLEAN for yes/no questions, or the INTEGER minute k of the earliest
delivery (NULL if impossible within the plan) for 'earliest' questions. Output only the SQL."""

VIEW_TXT = """Tables (DuckDB):
  node(nid INTEGER, name VARCHAR, kind VARCHAR)   -- kind in ('SAT','GS')
  vnode(nid INTEGER, k INTEGER)                    -- time-expanded node: node nid at minute k
  vedge(sn INTEGER, sk INTEGER, tn INTEGER, tk INTEGER, lab VARCHAR)
      -- edge (sn,sk) -> (tn,tk); lab = 'TX'  : transmission during minute k (sk = tk, sn -> tn)
      --                          lab = 'HOLD': the message stays at node sn from minute sk to sk+1
Times are minutes after 00:00 UTC (k = 60*hh + mm); the plan covers k = 0..179.
A message leaving node s at minute k0 can be at node d at minute k if and only if (d, k) is
reachable from (s, k0) by following vedge edges. The number of transmissions is the number of TX edges."""

IR_TASK_TXT = """Translate the operator question into a JSON object following this schema (output JSON only):
{schema}
Rules: task EXISTS for yes/no questions (arrive_by required), EARLIEST for 'earliest/soonest' questions.
Times as HH:MM UTC. Node names must be copied exactly from the list below.
Ground stations: {gs}
Satellites are named S<plane>-<index>, e.g. S03-07."""


def build_prompt(system, question, gs_names):
    if system == "ir":
        return IR_TASK_TXT.format(schema=json.dumps(IR_SCHEMA), gs=", ".join(gs_names)) + \
            f"\n\nQuestion: {question}\nJSON:"
    if system == "view_sql":
        return VIEW_TXT + "\n\n" + SQL_TASK_TXT + f"\n\nQuestion: {question}\nSQL:"
    p = SCHEMA_TXT + ("\n\n" + HINT_TXT if system == "sql_hint" else "") + "\n\n" + SQL_TASK_TXT
    return p + f"\n\nQuestion: {question}\nSQL:"


# ------------------------------------------------------------------ providers
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _post(url, headers, body, attempts=6):
    """POST with exponential backoff on rate limits, overload and transient errors."""
    for i in range(attempts):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=180)
            if r.status_code in RETRY_STATUS and i < attempts - 1:
                time.sleep(min(60, 2 ** i + random.random()))
                continue
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            if i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i + random.random()))


def call_anthropic(prompt, model, temperature=0.0, max_tokens=800):
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    j = _post("https://api.anthropic.com/v1/messages", {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"}, body)
    return "".join(b.get("text", "") for b in j["content"] if b.get("type") == "text")


def call_openai(prompt, model, base_url, temperature=0.0, max_tokens=800):
    base_url = base_url or "https://api.openai.com/v1"       # OpenAI itself unless another server is given
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    j = _post(base_url.rstrip("/") + "/chat/completions",
              {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"}, body)
    return j["choices"][0]["message"]["content"]


PROMPT_SHA = hashlib.sha256(json.dumps([SCHEMA_TXT, HINT_TXT, SQL_TASK_TXT, VIEW_TXT, IR_TASK_TXT,
                                        IR_SCHEMA]).encode()).hexdigest()[:12]


def naive_sql(names, item):
    g = item["gold_ir"]; idx = {n: i for i, n in enumerate(names)}
    s, d = idx[g["src"]], idx[g["dst"]]
    k0 = hhmm_to_slot(g["depart_after"])
    k1 = hhmm_to_slot(g["arrive_by"]) if g["arrive_by"] else 179
    core = f"""WITH RECURSIVE r(n) AS (SELECT {s} UNION SELECT c.v FROM r JOIN contact c ON c.u = r.n
               WHERE c.k BETWEEN {k0} AND {k1}) SELECT count(*) > 0 FROM r WHERE n = {d}"""
    if g["task"] == "EXISTS":
        return f"SELECT ({core}) AS answer"
    return f"SELECT CASE WHEN ({core}) THEN {k0} ELSE NULL END AS answer"


def get_output(provider, system, prompt, item, names, a):
    if provider == "anthropic":
        return call_anthropic(prompt, a.model, a.temperature)
    if provider == "openai":
        return call_openai(prompt, a.model, a.base_url, a.temperature)
    if provider == "mock-gold":
        ir = JourneyIR(**item["gold_ir"])
        return ir.to_json() if system == "ir" else compile_sql(names, ir, T, over_view=(system == "view_sql"))
    if provider == "mock-naive":
        return json.dumps(item["gold_ir"]) if system == "ir" else naive_sql(names, item)
    raise ValueError(provider)


# ------------------------------------------------------------------ execution
def run_sql(con, sql):
    sql = sql.strip().strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:]
    out = {}
    def work():
        try:
            out["v"] = con.execute(sql).fetchone()[0]
        except Exception as e:
            out["e"] = str(e)[:200]
    th = threading.Thread(target=work, daemon=True)
    th.start(); th.join(SQL_TIMEOUT_S)
    if th.is_alive():
        con.interrupt(); th.join(5)
        return None, "timeout"
    return out.get("v"), out.get("e")


def norm_answer(task, v):
    if task == "EXISTS":
        return None if v is None else bool(v)
    return None if v is None else int(v)


def stratified(items, limit):
    """First `limit` items taken round-robin over (regime, template) strata, so that a small dry
    run (e.g. --limit 20 -> 2 per stratum) exercises every query type and regime. No limit: all."""
    if not limit or limit >= len(items):
        return items
    strata = collections.OrderedDict()
    for it in items:
        strata.setdefault((it["regime"], it["template"]), []).append(it)
    out, i = [], 0
    while len(out) < limit:
        for key in strata:
            if i < len(strata[key]) and len(out) < limit:
                out.append(strata[key][i])
        i += 1
    return out


def load_env(reg, envs):
    if reg not in envs:
        cp = make_contact_plan(6, 11, T, reg)
        sv = SQLView(cp)
        sv.con.execute("CREATE TABLE IF NOT EXISTS node_names(nid INTEGER, name VARCHAR, kind VARCHAR)")
        sv.con.executemany("INSERT INTO node_names VALUES (?,?,?)",
                           [(i, DISPLAY.get(n, n), k) for i, (n, k) in enumerate(zip(cp.names, cp.kinds))])
        sv.con.execute("CREATE OR REPLACE TABLE node AS SELECT * FROM node_names")
        envs[reg] = (cp, Index(cp), sv)
    return envs[reg]


def score(it, raw, system, env):
    cp, ix, sv = env
    task = it["gold_ir"]["task"]
    rec = {}
    if system == "ir":
        try:
            ir = parse_ir(raw, cp.names)
            rec["ir"] = ir.canonical()
            rec["ir_exact"] = (ir.canonical() == it["gold_ir"])
            res = execute(ix, cp.names, ir)
            rec["certified"] = res.certified
            pred = res.answer if res.certified else "REJ"      # uncertified answers are withheld
        except (IRError, KeyError, ValueError, TypeError) as e:
            pred, rec["error"] = "ERR", f"ir: {e}"[:200]
    else:
        v, err = run_sql(sv.con, raw)
        if err:
            pred, rec["error"] = "ERR", err
        else:
            try:
                pred = norm_answer(task, v)
            except (ValueError, TypeError) as e:
                pred, rec["error"] = "ERR", f"type: {e}"[:200]
    rec["pred"], rec["gold"] = pred, it["gold_answer"]
    rec["released"] = pred not in ("ERR", "REJ")
    rec["correct"] = rec["released"] and (pred == it["gold_answer"])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--system", choices=["sql", "sql_hint", "view_sql", "ir"], required=True)
    ap.add_argument("--provider", default="mock-gold")
    ap.add_argument("--model"); ap.add_argument("--base-url")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature (use --temperature -1 to omit the parameter)")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls in the generation phase")
    ap.add_argument("--limit", type=int, help="stratified subset: round-robin over (regime, template)")
    ap.add_argument("--out", help="scored log (JSONL); raw generations are cached in <out>.raw.jsonl and resumed")
    ap.add_argument("--label", help="LLM label for tables, e.g. A or B")
    ap.add_argument("--emit-macro", help="write overall metrics as LaTeX macros with this prefix")
    a = ap.parse_args()
    if a.temperature is not None and a.temperature < 0:
        a.temperature = None
    if a.provider in ("anthropic", "openai") and not a.model:
        ap.error("--model is required for real providers")
    items = stratified([json.loads(l) for l in open(a.bench)], a.limit)
    envs = {}
    for reg in sorted({it["regime"] for it in items}):
        load_env(reg, envs)

    # ---- phase 1: generation (parallel, cached, resumable)
    raw_path = (a.out + ".raw.jsonl") if a.out else None
    raw = {}
    if raw_path and os.path.exists(raw_path):
        for l in open(raw_path):
            r = json.loads(l)
            if r.get("prompt_sha") == PROMPT_SHA and r.get("model") == a.model and r.get("system") == a.system:
                raw[r["id"]] = r
    todo = [it for it in items if it["id"] not in raw]
    lock = threading.Lock()
    fh = open(raw_path, "a") if raw_path else None

    def gen(it):
        cp = envs[it["regime"]][0]
        gs_names = [n for n, k in zip(cp.names, cp.kinds) if k == "GS"]
        prompt = build_prompt(a.system, it["question"], gs_names)
        t = time.perf_counter()
        out = get_output(a.provider, a.system, prompt, it, cp.names, a)
        return dict(id=it["id"], system=a.system, provider=a.provider, model=a.model, prompt_sha=PROMPT_SHA,
                    temperature=a.temperature, t_llm=time.perf_counter() - t, raw=out,
                    ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = {ex.submit(gen, it): it for it in todo}
        for n, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                r = f.result()
            except Exception as e:                      # keep going; the item can be resumed later
                failures += 1
                print(f"  generation failed for {it['id']}: {str(e)[:120]}", flush=True)
                continue
            raw[r["id"]] = r
            if fh:
                with lock:
                    fh.write(json.dumps(r) + "\n"); fh.flush()
            if n % 50 == 0:
                print(f"  generated {n}/{len(todo)}", flush=True)
    if fh:
        fh.close()

    # ---- phase 2: scoring (sequential, deterministic)
    stats = collections.Counter()
    per_t = collections.defaultdict(collections.Counter)
    logs = []
    for it in items:
        if it["id"] not in raw:
            continue
        r = raw[it["id"]]
        rec = dict(id=it["id"], instance=it.get("instance", it["id"]), template=it["template"],
                   regime=it["regime"], system=a.system, provider=r.get("provider", a.provider),
                   model=a.model, label=a.label,
                   prompt_sha=PROMPT_SHA, raw=r["raw"], t_llm=r["t_llm"])
        rec.update(score(it, r["raw"], a.system, envs[it["regime"]]))
        for c in (stats, per_t[it["template"]]):
            c["n"] += 1; c["correct"] += rec["correct"]; c["released"] += rec["released"]
            c["error"] += (not rec["released"]); c["ir_exact"] += rec.get("ir_exact", False)
            c["wrong_released"] += (rec["released"] and not rec["correct"])
        logs.append(rec)

    def fmt(c):
        n = max(c["n"], 1); r = max(c["released"], 1)
        return (f"acc {100*c['correct']/n:5.1f}%  coverage {100*c['released']/n:5.1f}%  "
                f"acc|released {100*c['correct']/r:5.1f}%  wrong-released {100*c['wrong_released']/n:5.1f}%  "
                f"rejected/err {100*c['error']/n:5.1f}%  IR-exact {100*c['ir_exact']/n:5.1f}%  (n={c['n']})")
    if a.emit_macro:
        d = os.path.join(os.environ.get("ORBITPROOF_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")), "gen")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{a.emit_macro}.tex"), "w") as fh:
            n = max(stats['n'], 1); r = max(stats['released'], 1); M = a.emit_macro
            fh.write(f"\\newcommand{{\\{M}}}{{{100*stats['correct']/n:.1f}}}\n")
            fh.write(f"\\newcommand{{\\{M}N}}{{{stats['n']}}}\n")
            fh.write(f"\\newcommand{{\\{M}Cov}}{{{100*stats['released']/n:.1f}}}\n")
            fh.write(f"\\newcommand{{\\{M}AccRel}}{{{100*stats['correct']/r:.1f}}}\n")
            fh.write(f"\\newcommand{{\\{M}WrongRel}}{{{100*stats['wrong_released']/n:.1f}}}\n")
            fh.write(f"\\newcommand{{\\{M}Rej}}{{{100*stats['error']/n:.1f}}}\n")
            fh.write(f"\\newcommand{{\\{M}IRx}}{{{100*stats['ir_exact']/n:.1f}}}\n")
            fh.write(f"\\newcommand{{\\{M}NCorrect}}{{{stats['correct']}}}\n")
            fh.write(f"\\newcommand{{\\{M}NWrong}}{{{stats['n'] - stats['correct']}}}\n")
    if a.out:
        with open(a.out, "w") as fh:
            for r in logs:
                fh.write(json.dumps(r, default=str) + "\n")
    print(f"[{a.system} / {a.provider} / {a.model}] prompts {PROMPT_SHA}  missing {len(items) - len(logs)}")
    print(f"   overall      {fmt(stats)}")
    for tpl, c in sorted(per_t.items()):
        print(f"   {tpl:12s} {fmt(c)}")
    if failures:
        print(f"{failures} generations failed; re-run the same command to resume.")


if __name__ == "__main__":
    main()
