"""Regression tests for the failures reported in the external pre-submission review.
Each case must now be rejected (or behave correctly). Exits non-zero on any failure."""
import json, sys
import numpy as np
from constellation import ContactPlan
from teg import INF, Index, check_answer, check_via_answer, earliest_arrival, verify_journey, verify_labeling
from agent import IRError, JourneyIR, Result, execute, parse_ir, render, verify_certificate

failures = []


def expect(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(name)


def plan(names, T, contacts, sort=True):
    if sort:
        contacts = sorted(contacts, key=lambda c: c[2])
    arr = np.array(contacts, dtype=np.int32).reshape(-1, 3)
    return Index(ContactPlan(names, ["GS"] * len(names), T, arr[:, 0], arr[:, 1], arr[:, 2]))


def rejects(fn):
    try:
        fn()
        return False
    except (IRError, TypeError, ValueError):
        return True


# 1. non-integer labels (NaN, floats) are outside the certificate domain
ix = plan(["S", "M", "D"], 3, [(0, 1, 0), (1, 2, 1)])
expect("NaN label rejected", not check_answer(ix, "EXISTS", 0, 2, 0, 2, False, None, [0, float("nan"), INF]))
expect("float label rejected", not verify_labeling(ix, [0, 1.0, INF], 0, 0, 2))
expect("wrong-length labeling rejected", not verify_labeling(ix, [0, 1], 0, 0, 2))
expect("bool claim for EARLIEST rejected", not check_answer(ix, "EARLIEST", 0, 2, 0, 2, False, None, [0, 0, 1]))

# 2. empty witnesses must satisfy via, avoidance and deadline
ix = plan(["S", "G"], 3, [])
expect("VIA with empty witness rejected", not check_via_answer(ix, "EXISTS", 0, 0, 1, 0, 2, True, [], [0, INF], None))
expect("avoided source rejected", not verify_journey(ix, [], 0, 0, 0, 2, avoid={0}))
expect("departure after deadline rejected", not verify_journey(ix, [], 0, 0, 2, 1))

# 3. out-of-horizon times are rejected, never clamped
ix = plan(["S", "D"], 3, [(0, 1, 2)])
expect("departure outside horizon rejected",
       rejects(lambda: execute(ix, ix.cp.names, parse_ir(json.dumps(
           {"task": "EARLIEST", "src": "S", "dst": "D", "depart_after": "00:10"}), ix.cp.names))))

# 4. schema violations are rejected
bad = [{"task": "EXISTS", "src": "S", "dst": "D", "depart_after": "00:00", "arrive_by": "00:02", "max_hops": 0},
       {"task": "EXISTS", "src": "S", "dst": "D", "depart_after": "00:00", "arrive_by": "00:02", "max_hops": True},
       {"task": "EXISTS", "src": "S", "dst": "D", "depart_after": "00:02", "arrive_by": "00:01"},
       {"task": "EXISTS", "src": "S", "dst": "D", "depart_after": "0:61", "arrive_by": "00:02"},
       {"task": "EXISTS", "src": "S", "dst": "D", "depart_after": "00:00", "arrive_by": "00:02", "note": "x"},
       {"task": "EXISTS", "src": "S", "dst": "D", "depart_after": "00:00", "arrive_by": "00:02", "avoid": "S"}]
for i, b in enumerate(bad):
    expect(f"invalid IR #{i + 1} rejected", rejects(lambda b=b: parse_ir(json.dumps(b), ix.cp.names)))

# 5. contact row order does not matter (plan is canonicalised before indexing)
ix = plan(["S", "D"], 2, [(0, 1, 1), (1, 0, 0)], sort=False)
truth = earliest_arrival(ix, 0, 1, 1)[0][1]
expect("false NO on unsorted contacts rejected",
       truth == 1 and not check_answer(ix, "EXISTS", 0, 1, 1, 1, False, None, [1, INF]))

# 6. VIA: answer certified and correct (subset-minimality is only claimed for node-simple journeys)
names = ["S", "A", "G", "D"]
ix = plan(names, 3, [(0, 1, 0), (1, 2, 1), (2, 0, 1), (0, 1, 1), (1, 3, 1)])
res = execute(ix, names, JourneyIR("EARLIEST", "S", "D", "00:00", via="G"))
expect("VIA answer certified", res.certified and res.answer == 1)

# 7. renderer: refusal on failed certification, full IR echo on success
ir = JourneyIR("EXISTS", "S", "D", "00:00", "00:02")
expect("renderer refuses uncertified answer", render(names, ir, Result(True, [], False)).startswith("Answer withheld"))
ok = execute(ix, names, JourneyIR("EXISTS", "S", "D", "00:00", "00:02", avoid=["G"]))
txt = render(names, JourneyIR("EXISTS", "S", "D", "00:00", "00:02", avoid=["G"]), ok)
expect("renderer echoes every constraint", "avoiding G" in txt and "arriving by 00:02" in txt, txt)

# 8. exported certificates are checked against the plan snapshot only
cert = res.detail["certificate"]
expect("exported certificate accepted", verify_certificate(ix, names, cert))
for key, val in (("answer", 0), ("plan_digest", "0" * 64), ("labeling_via", [0, 0, 0, 0]), ("tau", 1)):
    expect(f"tampered certificate ({key}) rejected", not verify_certificate(ix, names, {**cert, key: val}))

# 9. scorer: exact answer types
try:
    from harness import norm_answer
    for task, v in (("EXISTS", "false"), ("EARLIEST", False), ("EARLIEST", 1.9), ("EXISTS", None)):
        expect(f"scorer rejects {task}={v!r}", rejects(lambda task=task, v=v: norm_answer(task, v)))
except ImportError as e:
    print("SKIP scorer checks:", e)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
