import time, random
from constellation import make_contact_plan
from teg import *
from enum_baseline import enumerate_paths
random.seed(2)
cp = make_contact_plan(6, 11, 180, "no_isl"); ix = Index(cp)
gs = list(range(66, cp.N))
# 1) hop-bounded TEG vs exhaustive pruned enumeration (small H, correctness)
mism=0; tot=0
for q in range(40):
    s,d = random.sample(gs,2); k0=random.randint(0,150); k1=k0+20
    for H in (1,2,3,4):
        f,par,vis = earliest_arrival_hops(ix,s,k0,k1,H)
        teg = min(f[d]) < INF
        found, exp, cap = enumerate_paths(ix,s,d,k0,k1,H,"pruned",cap=5*10**6)
        if cap: continue
        tot+=1; mism += (teg!=found)
        if teg:
            J = journey_from_parent_hops(par,f,s,d,H)
            assert verify_journey(ix,J,s,d,k0,k1,H=H), (J)
print("hop-bounded agreement mismatches", mism, "of", tot)
# 2) loop elimination on random journeys with loops
cnt=0
for q in range(300):
    s = random.choice(gs); k0=random.randint(0,100)
    J=[]; a=s; k=k0
    for step in range(12):
        opts=[(a2,b2,kk) for kk in range(k, min(k+5,cp.T)) for b2 in ix.adj[kk].get(a,[]) for a2 in [a]]
        if not opts: break
        c=random.choice(opts); J.append(c); a=c[1]; k=c[2]
    if not J: continue
    d = J[-1][1]
    if d==s: continue
    J2 = loop_eliminate(J, s)
    assert verify_journey(ix,J2,s,d,k0,10**6), (J,J2)
    assert is_node_simple(J2,s)
    assert J2[-1][2] <= J[-1][2]
    assert set(J2) <= set(J)
    cnt+=1
print("loop-elim ok on", cnt)
# 3) tampered certificates are rejected
rej=0; tot=0
for q in range(100):
    s,d = random.sample(gs,2); k0=random.randint(0,120); k1=k0+60
    f,par,_ = earliest_arrival(ix,s,k0,k1)
    if f[d] < INF and f[d] > k0:
        g=list(f); g[d]=f[d]+1  # claim later earliest arrival than truth
        tot+=1; rej += (not verify_labeling(ix,g,s,k0,k1))
        g=list(f); g[d]=INF     # claim unreachable
        tot+=1; rej += (not verify_labeling(ix,g,s,k0,k1))
print("tampered labelings rejected", rej, "of", tot)
