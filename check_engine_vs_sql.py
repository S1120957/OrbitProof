import time, random
from constellation import make_contact_plan
from teg import *
from sqlview import SQLView
random.seed(1)
cp = make_contact_plan(6, 11, 180, "isl_intra")
ix = Index(cp)
t=time.time(); sv = SQLView(cp); print("view", sv.materialise_s, sv.n_vnode, sv.n_vedge, time.time()-t)
nsat = 66
gs = list(range(nsat, cp.N))
agree=0; n=0; tsql=[]; tpy=[]
for q in range(30):
    s, d = random.sample(gs, 2); k0 = random.randint(0, 120); k1 = k0+60
    t=time.perf_counter(); f, par, vis = earliest_arrival(ix, s, k0, k1); tpy.append(time.perf_counter()-t)
    t=time.perf_counter(); r = sv.earliest(s, d, k0, k1); tsql.append(time.perf_counter()-t)
    py = f[d] if f[d] < INF else None
    agree += (py == r); n+=1
    if py is not None:
        J = journey_from_parent(par, s, d)
        assert verify_journey(ix, J, s, d, k0, k1), "verify"
        assert is_node_simple(J, s)
        assert replay_earliest(ix, J, s, d, k0) == py
        assert witness_is_minimal(ix, J, s, d, k0)
    assert verify_labeling(ix, f, s, k0, k1)
print("agree", agree, n, "py ms", 1000*sum(tpy)/n, "sql ms", 1000*sum(tsql)/n)
