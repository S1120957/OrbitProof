"""Certifying query pipeline:  NL --(LLM)--> Journey IR --(compiler)--> TEG view query
                               --(engine)--> answer + certificate --(verifier)--> response

The LLM only produces the IR (intent + entity grounding). The composite-identifier
encoding lives in the deterministic compiler, and every answer is released only
with a certificate that an independent checker has accepted:
  positive answers  -> journey witness J        (verify_journey,  O(|J|))
  negative/optimal  -> closed labeling f        (verify_labeling, O(#contacts in window))
"""
from __future__ import annotations
import json, re
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
        return d


class IRError(ValueError):
    pass


IR_FIELDS = {"task", "src", "dst", "depart_after", "arrive_by", "avoid", "via", "max_hops"}
_HHMM = re.compile(r"^([01][0-9]|2[0-3]):[0-5][0-9]$")


def parse_ir(text, names):
    """Parse, validate against the IR schema, and ground an LLM-produced IR. Raises IRError.
    Nothing is silently dropped or coerced: unknown fields, wrong types, malformed times,
    max_hops < 1 and unsupported combinations are rejected."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    try:
        obj = json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception as e:
        raise IRError(f"not JSON: {e}")
    if not isinstance(obj, dict):
        raise IRError("IR must be a JSON object")
    extra = set(obj) - IR_FIELDS
    if extra:
        raise IRError(f"unknown fields {sorted(extra)}")
    for k in ("task", "src", "dst", "depart_after"):
        if k not in obj:
            raise IRError(f"missing field {k!r}")
    lut = {n.lower(): n for n in names}

    def ground(x, what):
        if not isinstance(x, str):
            raise IRError(f"{what} must be a string")
        k = x.strip().lower().replace(" ", "")
        if k not in lut:
            raise IRError(f"unknown node {x!r}")
        return lut[k]

    def hhmm(x, what):
        if not isinstance(x, str) or not _HHMM.match(x):
            raise IRError(f"{what} must be HH:MM, got {x!r}")
        return x

    task = obj["task"]
    if task not in ("EXISTS", "EARLIEST"):
        raise IRError("task must be EXISTS or EARLIEST")
    avoid = obj.get("avoid") or []
    if not isinstance(avoid, list):
        raise IRError("avoid must be a list")
    via = obj.get("via")
    hops = obj.get("max_hops")
    if hops is not None and (isinstance(hops, bool) or not isinstance(hops, int) or hops < 1):
        raise IRError("max_hops must be an integer >= 1 or null")
    arrive = obj.get("arrive_by")
    ir = JourneyIR(task=task, src=ground(obj["src"], "src"), dst=ground(obj["dst"], "dst"),
                   depart_after=hhmm(obj["depart_after"], "depart_after"),
                   arrive_by=None if arrive is None else hhmm(arrive, "arrive_by"),
                   avoid=[ground(a, "avoid entry") for a in avoid],
                   via=None if via is None else ground(via, "via"), max_hops=hops)
    if ir.task == "EXISTS" and ir.arrive_by is None:
        raise IRError("EXISTS needs arrive_by")
    if ir.via is not None and ir.max_hops is not None:
        raise IRError("via + max_hops is not supported by this IR version")
    if ir.arrive_by is not None and hhmm_to_slot(ir.arrive_by) < hhmm_to_slot(ir.depart_after):
        raise IRError("arrive_by is before depart_after")
    return ir


def ir_window(ir, T, dt=60):
    """Map the IR's times to plan slots. Times outside the plan horizon are rejected,
    never clamped, so the executed query is exactly the validated IR."""
    def slot(hhmm):
        sec = hhmm_to_slot(hhmm) * 60
        if sec % dt:
            raise IRError(f"time {hhmm} is not aligned to the {dt}-s slots of the plan")
        return sec // dt
    k0 = slot(ir.depart_after)
    k1 = slot(ir.arrive_by) if ir.arrive_by is not None else T - 1
    if not 0 <= k0 <= T - 1 or not 0 <= k1 <= T - 1:
        raise IRError(f"time outside the plan horizon (slots 0..{T - 1})")
    if k1 < k0:
        raise IRError("deadline before departure")
    return k0, k1


@dataclass
class Result:
    answer: object            # True/False for EXISTS, slot or None for EARLIEST
    journey: Optional[list]   # witness (list of contacts) if one exists
    certified: bool
    detail: dict = field(default_factory=dict)   # detail["certificate"]: self-contained artifact


def execute(ix, names, ir: JourneyIR) -> Result:
    """Run the validated IR on the engine and build a self-contained certificate; the answer is
    certified only if verify_certificate() accepts that certificate against the base plan."""
    if list(names) != list(ix.cp.names):
        raise IRError("node names do not match the plan")
    idx = {n: i for i, n in enumerate(names)}
    s, d = idx[ir.src], idx[ir.dst]
    k0, k1 = ir_window(ir, ix.T, ix.dt)
    avoid = frozenset(idx[a] for a in ir.avoid)
    J = f = f2 = None
    kd = None
    if ir.via is not None:
        g = idx[ir.via]
        f1, p1, _ = earliest_arrival(ix, s, k0, k1, avoid)
        f = f1
        if f1[g] < INF:
            f2, p2, _ = earliest_arrival(ix, g, f1[g], k1, avoid)
            if f2[d] < INF:
                kd = f2[d]
                J = journey_from_parent(p1, s, g) + journey_from_parent(p2, g, d)
    elif ir.max_hops is not None:
        f, par, _ = earliest_arrival_hops(ix, s, k0, k1, ir.max_hops, avoid)
        if min(f[d]) < INF:
            kd = min(f[d])
            J = loop_eliminate(journey_from_parent_hops(par, f, s, d, ir.max_hops), s)
    else:
        f, par, _ = earliest_arrival(ix, s, k0, k1, avoid)
        if f[d] < INF:
            kd = f[d]
            J = journey_from_parent(par, s, d)
    answer = (kd is not None) if ir.task == "EXISTS" else kd
    cert = dict(version=1, plan_digest=ix.digest, slot_s=ix.dt, tau=0, ir=ir.canonical(), answer=answer,
                journey=[list(c) for c in J] if J is not None else None,
                labeling=[list(r) for r in f] if ir.max_hops is not None else list(f),
                labeling_via=list(f2) if f2 is not None else None)
    ok = verify_certificate(ix, names, cert)
    return Result(answer, J, ok, {"certificate": cert})


def verify_certificate(ix, names, cert):
    """Independent check of a certificate against the base plan only: the IR is re-validated,
    the plan digest must match, and the answer-level checkers of teg.py decide."""
    try:
        if not isinstance(cert, dict) or cert.get("version") != 1 or cert.get("plan_digest") != ix.digest:
            return False
        if cert.get("slot_s") != ix.dt or cert.get("tau") != 0:
            return False
        ir = parse_ir(json.dumps(cert["ir"]), names)
        idx = {n: i for i, n in enumerate(names)}
        s, d = idx[ir.src], idx[ir.dst]
        k0, k1 = ir_window(ir, ix.T, ix.dt)
        avoid = frozenset(idx[a] for a in ir.avoid)
        J, f, claim = cert["journey"], cert["labeling"], cert["answer"]
        if J is not None:
            J = [tuple(c) for c in J]
        if ir.via is not None:
            return bool(check_via_answer(ix, ir.task, s, d, idx[ir.via], k0, k1, claim, J, f,
                                         cert.get("labeling_via"), avoid))
        if ir.max_hops is not None:
            return bool(check_hops_answer(ix, ir.task, s, d, k0, k1, ir.max_hops, claim, J, f, avoid))
        return bool(check_answer(ix, ir.task, s, d, k0, k1, claim, J, f, avoid))
    except (IRError, KeyError, TypeError, ValueError, IndexError):
        return False


def describe_ir(ir):
    """The interpreted query in words, echoed to the operator with every answer."""
    parts = [("Is there a delivery" if ir.task == "EXISTS" else "Earliest delivery"),
             f"from {ir.src} to {ir.dst}", f"departing at or after {ir.depart_after} UTC"]
    if ir.arrive_by is not None:
        parts.append(f"arriving by {ir.arrive_by} UTC")
    if ir.avoid:
        parts.append("avoiding " + ", ".join(ir.avoid))
    if ir.via is not None:
        parts.append(f"via {ir.via}")
    if ir.max_hops is not None:
        parts.append(f"with at most {ir.max_hops} transmissions")
    return " ".join(parts)


def render(names, ir, res: Result):
    """Operator-facing text generated ONLY from a certified result; otherwise a refusal."""
    if not res.certified:
        return "Answer withheld: the certificate was not accepted. Interpreted as: " + describe_ir(ir) + "."
    if ir.task == "EXISTS":
        head = "Yes" if res.answer else "No"
    else:
        head = (f"Earliest delivery {slot_to_hhmm(res.answer)} UTC" if res.answer is not None
                else "Not deliverable within the plan horizon")
    text = f"{head}. Interpreted as: {describe_ir(ir)}."
    if res.journey:
        text += " Witness: " + "; ".join(f"{names[a]}->{names[b]} @{slot_to_hhmm(k)}" for a, b, k in res.journey)
    return text


# ----------------------------------------------------------------------------
# IR -> SQL over the base tables (view defined inline). Used by the prototype and
# to test the compiler against the engine.
# ----------------------------------------------------------------------------
def compile_sql(names, ir: JourneyIR, T, over_view=False):
    idx = {n: i for i, n in enumerate(names)}
    s, d = idx[ir.src], idx[ir.dst]
    k0, k1 = ir_window(ir, T)
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
    if ir.max_hops is not None:
        body = f"""
  reach(n, k, h) AS (
    SELECT {s}, {k0}, 0
    UNION
    SELECT e.tn, e.tk, r.h + e.dh FROM reach r JOIN vw e ON e.sn = r.n AND e.sk = r.k
     WHERE r.h + e.dh <= {ir.max_hops})
SELECT min(k) FROM reach WHERE n = {d}"""
    elif ir.via is not None:
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
