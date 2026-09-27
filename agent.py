"""Certifying query pipeline:  NL --(LLM)--> Journey IR --(compiler)--> TEG view query
                               --(engine)--> answer + certificate --(verifier)--> response

The LLM only produces the IR (intent + entity grounding). The composite-identifier
encoding lives in the deterministic compiler, and every answer is released only
with a certificate that an independent checker has accepted:
  positive answers  -> journey witness J        (verify_journey,  O(|J|))
  negative/optimal  -> closed labeling f        (verify_labeling, O(#contacts in window))
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from teg import (INF, earliest_arrival, earliest_arrival_hops, journey_from_parent, check_answer,
                 journey_from_parent_hops, loop_eliminate, verify_journey, verify_labeling,
                 verify_labeling_hops, check_via_answer, check_hops_answer, arrival)

IR_SCHEMA = {
    "type": "object",
    "required": ["task", "src", "dst", "depart_after"],
    "properties": {
        "task": {"enum": ["EXISTS", "EARLIEST"]},
        "src": {"type": "string"}, "dst": {"type": "string"},
        "depart_after": {"type": "string", "pattern": "^[0-2][0-9]:[0-5][0-9]$"},
        "arrive_by": {"type": ["string", "null"]},
        "avoid": {"type": "array", "items": {"type": "string"}},
        "via": {"type": ["string", "null"]},
        "max_hops": {"type": ["integer", "null"], "minimum": 1},
    },
}


def hhmm_to_slot(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def slot_to_hhmm(k):
    return f"{k // 60:02d}:{k % 60:02d}"


@dataclass
class JourneyIR:
    task: str
    src: str
    dst: str
    depart_after: str
    arrive_by: Optional[str] = None
    avoid: list = field(default_factory=list)
    via: Optional[str] = None
    max_hops: Optional[int] = None

    def to_json(self):
        return json.dumps(asdict(self))

    def canonical(self):
        d = asdict(self)
        d["avoid"] = sorted(d["avoid"] or [])
        if d["task"] == "EARLIEST":
            d["arrive_by"] = None
        return d


class IRError(ValueError):
    pass


def parse_ir(text, names):
    """Parse + validate + ground an LLM-produced IR. Raises IRError."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception as e:
        raise IRError(f"not JSON: {e}")
    lut = {n.lower(): n for n in names}
    def ground(x):
        if x is None:
            return None
        k = str(x).strip().lower().replace(" ", "")
        if k not in lut:
            raise IRError(f"unknown node {x!r}")
        return lut[k]
    try:
        ir = JourneyIR(task=obj["task"], src=ground(obj["src"]), dst=ground(obj["dst"]),
                       depart_after=obj["depart_after"], arrive_by=obj.get("arrive_by"),
                       avoid=[ground(a) for a in (obj.get("avoid") or [])],
                       via=ground(obj.get("via")), max_hops=obj.get("max_hops"))
    except KeyError as e:
        raise IRError(f"missing field {e}")
    if ir.task not in ("EXISTS", "EARLIEST"):
        raise IRError("bad task")
    if ir.task == "EXISTS" and not ir.arrive_by:
        raise IRError("EXISTS needs arrive_by")
    if ir.via and ir.max_hops:
        raise IRError("via + max_hops not supported by this IR version")
    for f_ in (ir.depart_after, ir.arrive_by):
        if f_ is not None:
            try:
                hhmm_to_slot(f_)
            except Exception:
                raise IRError(f"bad time {f_!r}")
    return ir


@dataclass
class Result:
    answer: object            # True/False for EXISTS, slot or None for EARLIEST
    journey: Optional[list]   # witness (list of contacts) if reachable
    certified: bool
    detail: dict = field(default_factory=dict)


def execute(ix, names, ir: JourneyIR) -> Result:
    idx = {n: i for i, n in enumerate(names)}
    s, d = idx[ir.src], idx[ir.dst]
    k0 = hhmm_to_slot(ir.depart_after)
    k1 = hhmm_to_slot(ir.arrive_by) if (ir.task == "EXISTS" and ir.arrive_by) else ix.T - 1
    k0 = max(0, min(k0, ix.T - 1)); k1 = min(k1, ix.T - 1)
    avoid = frozenset(idx[a] for a in ir.avoid)
    if ir.via:
        g = idx[ir.via]
        f1, p1, _ = earliest_arrival(ix, s, k0, k1, avoid)
        f2 = J = None
        kd = None
        if f1[g] < INF:
            f2, p2, _ = earliest_arrival(ix, g, f1[g], k1, avoid)
            if f2[d] < INF:
                kd = f2[d]
                J = journey_from_parent(p1, s, g) + journey_from_parent(p2, g, d)
        claim = (kd is not None) if ir.task == "EXISTS" else kd
        ok = check_via_answer(ix, ir.task, s, d, g, k0, k1, claim, J, f1, f2, avoid)
        return _pack(ir, kd, J, ok)
    if ir.max_hops:
        f, par, _ = earliest_arrival_hops(ix, s, k0, k1, ir.max_hops, avoid)
        kd = min(f[d]) if min(f[d]) < INF else None
        J = loop_eliminate(journey_from_parent_hops(par, f, s, d, ir.max_hops), s) if kd is not None else None
        claim = (kd is not None) if ir.task == "EXISTS" else kd
        ok = check_hops_answer(ix, ir.task, s, d, k0, k1, ir.max_hops, claim, J, f, avoid)
        return _pack(ir, kd, J, ok)
    f, par, _ = earliest_arrival(ix, s, k0, k1, avoid)
    J = journey_from_parent(par, s, d) if f[d] < INF else None
    kd = f[d] if f[d] < INF else None
    claim = (kd is not None) if ir.task == "EXISTS" else kd
    ok = check_answer(ix, ir.task, s, d, k0, k1, claim, J, f, avoid)   # answer-level certificate check
    return _pack(ir, kd, J, ok)


def _pack(ir, kd, J, ok, note=None):
    ans = (kd is not None) if ir.task == "EXISTS" else kd
    return Result(ans, J, bool(ok), {"note": note} if note else {})


def render(names, ir, res: Result):
    """Operator-facing text generated ONLY from the verified result (no free-form rationale)."""
    if ir.task == "EXISTS":
        head = "Yes" if res.answer else "No"
    else:
        head = f"Earliest delivery {slot_to_hhmm(res.answer)} UTC" if res.answer is not None else "Not deliverable in the plan horizon"
    if res.journey:
        hops = "; ".join(f"{names[a]}->{names[b]} @{slot_to_hhmm(k)}" for a, b, k in res.journey)
        head += f". Witness: {hops}"
    return head + ("" if res.certified else " [UNVERIFIED - withheld]")


# ----------------------------------------------------------------------------
# IR -> SQL over the base tables (view defined inline). Used by the prototype and
# to test the compiler against the engine.
# ----------------------------------------------------------------------------
def compile_sql(names, ir: JourneyIR, T, over_view=False):
    idx = {n: i for i, n in enumerate(names)}
    s, d = idx[ir.src], idx[ir.dst]
    k0 = max(0, min(hhmm_to_slot(ir.depart_after), T - 1))
    k1 = min(hhmm_to_slot(ir.arrive_by), T - 1) if (ir.task == "EXISTS" and ir.arrive_by) else T - 1
    av = ",".join(str(idx[a]) for a in ir.avoid) or "-1"
    if over_view:      # query the materialised TEG view tables vnode / vedge directly
        base = f"""
  vw AS (
    SELECT sn, sk, tn, tk, CASE WHEN lab = 'TX' THEN 1 ELSE 0 END AS dh FROM vedge
     WHERE sn NOT IN ({av}) AND tn NOT IN ({av}) AND sk >= {k0} AND tk <= {k1})"""
    else:              # define the view inline over the base tables
        base = f"""
  vw AS (
    SELECT c.u AS sn, c.k AS sk, c.v AS tn, c.k AS tk, 1 AS dh FROM contact c
     WHERE c.u NOT IN ({av}) AND c.v NOT IN ({av}) AND c.k BETWEEN {k0} AND {k1}
    UNION ALL
    SELECT n.nid, s.k, n.nid, s.nk, 0 FROM node n JOIN slot s ON s.nk IS NOT NULL
     WHERE n.nid NOT IN ({av}) AND s.k >= {k0} AND s.nk <= {k1})"""
    if ir.max_hops:
        body = f"""
  reach(n, k, h) AS (
    SELECT {s}, {k0}, 0
    UNION
    SELECT e.tn, e.tk, r.h + e.dh FROM reach r JOIN vw e ON e.sn = r.n AND e.sk = r.k
     WHERE r.h + e.dh <= {ir.max_hops})
SELECT min(k) FROM reach WHERE n = {d}"""
    elif ir.via:
        g = idx[ir.via]
        body = f"""
  r1(n, k) AS (
    SELECT {s}, {k0}
    UNION
    SELECT e.tn, e.tk FROM r1 JOIN vw e ON e.sn = r1.n AND e.sk = r1.k),
  r2(n, k) AS (
    SELECT {g}, (SELECT min(k) FROM r1 WHERE n = {g}) WHERE EXISTS (SELECT 1 FROM r1 WHERE n = {g})
    UNION
    SELECT e.tn, e.tk FROM r2 JOIN vw e ON e.sn = r2.n AND e.sk = r2.k)
SELECT min(k) FROM r2 WHERE n = {d}"""
    else:
        body = f"""
  reach(n, k) AS (
    SELECT {s}, {k0}
    UNION
    SELECT e.tn, e.tk FROM reach r JOIN vw e ON e.sn = r.n AND e.sk = r.k)
SELECT min(k) FROM reach WHERE n = {d}"""
    sql = "WITH RECURSIVE" + base + "," + body
    if ir.task == "EXISTS":
        sql = f"SELECT (x IS NOT NULL) AS answer FROM ({sql}) AS t(x)"
    else:
        sql = f"SELECT x AS answer FROM ({sql}) AS t(x)"
    return sql
