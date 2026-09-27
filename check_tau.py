"""Cross-check tau in {0,1}: Python engine vs DuckDB view; witnesses, checker, replay, minimality."""
import random
from constellation import make_contact_plan
from teg import *
from sqlview import SQLView
rng = random.Random(3)
for reg in ("no_isl", "isl_intra"):
    cp = make_contact_plan(6, 11, 180, reg); ix = Index(cp)
    gs = [i for i, t in enumerate(cp.kinds) if t == "GS"]
    for tau in (0, 1):
        sv = SQLView(cp, tau=tau); agree = ok = n = 0
        for q in range(40):
            s, d = rng.sample(gs, 2); k0 = rng.randint(0, 110); k1 = k0 + 60
            f, par, _ = earliest_arrival(ix, s, k0, k1, tau=tau)
            py = f[d] if f[d] < INF else None
            agree += (py == sv.earliest(s, d, k0, k1)); n += 1
            if py is None:
                ok += check_answer(ix, "EARLIEST", s, d, k0, k1, None, None, f, tau=tau)
            else:
                J = journey_from_parent(par, s, d)
                f2, _, _ = earliest_arrival(ix, s, k0, k1, tau=tau, stop_at=d)   # early-exit labeling
                ok += (check_answer(ix, "EARLIEST", s, d, k0, k1, py, J, f2, tau=tau)
                       and replay_earliest(ix, J, s, d, k0, tau=tau) == py
                       and witness_is_minimal(ix, J, s, d, k0, tau=tau) and is_node_simple(J, s))
        print(f"{reg} tau={tau}: engine==DuckDB {agree}/{n}, certified+replay+minimal {ok}/{n}")
