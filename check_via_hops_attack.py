"""Regression test for the VIA/HOPS earliest-arrival attainment gap found in the internal audit.
A closed labeling that is lowered to (claim) stays closed, so labeling + valid journey alone would
accept a claim the journey does not attain. The answer-level checkers must reject it."""
import json
from constellation import make_contact_plan
from teg import *
from agent import JourneyIR, hhmm_to_slot

item = next(json.loads(l) for l in open("bench.jsonl") if json.loads(l)["id"] == "isl_intra-T4_via-005-p0")
ir = JourneyIR(**item["gold_ir"])
cp = make_contact_plan(6, 11, 180, item["regime"]); ix = Index(cp)
idx = {n: i for i, n in enumerate(cp.names)}
s, d, g, k0, k1 = idx[ir.src], idx[ir.dst], idx[ir.via], hhmm_to_slot(ir.depart_after), cp.T - 1
f1, p1, _ = earliest_arrival(ix, s, k0, k1); kg = f1[g]
f2, p2, _ = earliest_arrival(ix, g, kg, k1)
J = journey_from_parent(p1, s, g) + journey_from_parent(p2, g, d)
true = f2[d]
claim = true - 1
forged = [min(x, claim) for x in f2]                       # closed, but not attained
old_accepts = verify_labeling(ix, f1, s, k0, k1) and verify_labeling(ix, forged, g, kg, k1) \
    and verify_journey(ix, J, s, d, k0, k1, via=g)
new_accepts = check_via_answer(ix, "EARLIEST", s, d, g, k0, k1, claim, J, f1, forged)
print(f"VIA  {item['id']}: true arrival {true}, kg {kg}, forged claim {claim}: "
      f"old predicates accept={old_accepts}, new checker accepts={new_accepts}, "
      f"true answer accepted={check_via_answer(ix, 'EARLIEST', s, d, g, k0, k1, true, J, f1, f2)}")
assert old_accepts and not new_accepts

# HOPS: EARLIEST + max_hops
cp = make_contact_plan(6, 11, 180, "no_isl"); ix = Index(cp)
gs = [i for i, t in enumerate(cp.kinds) if t == "GS"]
import random; rng = random.Random(1); found = 0
for _ in range(200):
    s, d = rng.sample(gs, 2); k0 = rng.randint(0, 100); H = 3
    f, par, _ = earliest_arrival_hops(ix, s, k0, cp.T - 1, H)
    kd = min(f[d])
    if kd >= INF or kd <= k0:
        continue
    J = loop_eliminate(journey_from_parent_hops(par, f, s, d, H), s)
    forged = [[min(x, kd - 1) for x in row] for row in f]
    old = verify_labeling_hops(ix, forged, s, k0, cp.T - 1, H) and verify_journey(ix, J, s, d, k0, cp.T - 1, H=H)
    new = check_hops_answer(ix, "EARLIEST", s, d, k0, cp.T - 1, H, kd - 1, J, forged)
    ok = check_hops_answer(ix, "EARLIEST", s, d, k0, cp.T - 1, H, kd, J, f)
    assert old and not new and ok
    found += 1
print(f"HOPS: {found} EARLIEST+HOPS instances: forged lower bound accepted by old predicates, "
      f"rejected by the new checker; true answers accepted")
