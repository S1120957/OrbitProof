from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

R_E = 6371.0            # km
MU = 398600.4418        # km^3/s^2
OMEGA_E = 7.2921159e-5  # rad/s
DT = 60.0               # slot length [s]
ELEV_MIN_DEG = 10.0     # GS elevation mask
ISL_RANGE_KM = 6000.0   # max inter-plane ISL range
LAT_OFF_DEG = 60.0      # inter-plane ISLs switched off above this latitude

GROUND_STATIONS = [
    ("Fucino", 41.98, 13.60), ("London", 51.51, -0.13), ("NewYork", 40.71, -74.01),
    ("SaoPaulo", -23.55, -46.63), ("Nairobi", -1.29, 36.82), ("Tokyo", 35.68, 139.69),
    ("Sydney", -33.87, 151.21), ("Singapore", 1.35, 103.82), ("Mumbai", 19.08, 72.88),
    ("CapeTown", -33.92, 18.42), ("Svalbard", 78.23, 15.41), ("Santiago", -33.45, -70.67),
    ("Anchorage", 61.22, -149.90), ("Honolulu", 21.31, -157.86), ("Dubai", 25.20, 55.27),
    ("Lagos", 6.52, 3.38), ("LosAngeles", 34.05, -118.24), ("Reykjavik", 64.15, -21.94),
    ("Perth", -31.95, 115.86), ("Beijing", 39.90, 116.40),
]


@dataclass
class ContactPlan:
    names: list            # node names; satellites first, then ground stations
    kinds: list            # "SAT" / "GS"
    T: int                 # number of slots
    u: np.ndarray          # contact source   (int32)
    v: np.ndarray          # contact target   (int32)
    k: np.ndarray          # contact slot     (int32)
    meta: dict = field(default_factory=dict)

    @property
    def N(self):
        return len(self.names)

    @property
    def n_contacts(self):
        return int(self.u.shape[0])


def walker_star(P, S, alt_km=780.0, inc_deg=86.4, F=1):
    a = R_E + alt_km
    n = np.sqrt(MU / a**3)
    raan = np.array([p * np.pi / P for p in range(P) for _ in range(S)])
    m0 = np.array([2 * np.pi * s / S + 2 * np.pi * F * p / (P * S)
                   for p in range(P) for s in range(S)])
    plane = np.array([p for p in range(P) for _ in range(S)])
    idx = np.array([s for _ in range(P) for s in range(S)])
    return dict(a=a, n=n, inc=np.radians(inc_deg), raan=raan, m0=m0,
                plane=plane, idx=idx, P=P, S=S, alt=alt_km)


def sat_eci(c, t):
    u = c["m0"] + c["n"] * t
    cu, su = np.cos(u), np.sin(u)
    cO, sO = np.cos(c["raan"]), np.sin(c["raan"])
    ci, si = np.cos(c["inc"]), np.sin(c["inc"])
    x = cu * cO - su * ci * sO
    y = cu * sO + su * ci * cO
    z = su * si
    return c["a"] * np.stack([x, y, z], axis=1)


def gs_eci(t, stations=GROUND_STATIONS):
    lat = np.radians([s[1] for s in stations])
    lon = np.radians([s[2] for s in stations]) + OMEGA_E * t
    return R_E * np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def _links_at(c, t, regime):
    rs = sat_eci(c, t)
    rg = gs_eci(t)
    P, S = c["P"], c["S"]
    nsat = P * S
    pairs = []
    if regime in ("isl_full", "isl_intra"):
        for p in range(P):
            for s in range(S):
                i = p * S + s
                j = p * S + (s + 1) % S
                pairs.append((i, j))
    if regime == "isl_full":
        lat = np.degrees(np.arcsin(rs[:, 2] / np.linalg.norm(rs, axis=1)))
        for p in range(P - 1):              # no seam link P-1 <-> 0 (Walker star)
            for s in range(S):
                i, j = p * S + s, (p + 1) * S + s
                if abs(lat[i]) <= LAT_OFF_DEG and abs(lat[j]) <= LAT_OFF_DEG \
                        and np.linalg.norm(rs[i] - rs[j]) <= ISL_RANGE_KM:
                    pairs.append((i, j))
    # ground-satellite visibility
    rel = rs[None, :, :] - rg[:, None, :]                      # (G, nsat, 3)
    up = rg / np.linalg.norm(rg, axis=1, keepdims=True)
    sin_el = np.einsum("gsk,gk->gs", rel, up) / np.linalg.norm(rel, axis=2)
    vis = sin_el >= np.sin(np.radians(ELEV_MIN_DEG))
    g_idx, s_idx = np.nonzero(vis)
    for g, s in zip(g_idx, s_idx):
        pairs.append((int(s), nsat + int(g)))
    return set(pairs)


def make_contact_plan(P, S, T, regime, alt_km=780.0, inc_deg=86.4, t0=0.0, dt=DT):
    c = walker_star(P, S, alt_km, inc_deg)
    names = [f"S{p:02d}-{s:02d}" for p in range(P) for s in range(S)] + [g[0] for g in GROUND_STATIONS]
    kinds = ["SAT"] * (P * S) + ["GS"] * len(GROUND_STATIONS)
    links = [_links_at(c, t0 + k * dt, regime) for k in range(T + 1)]
    U, V, K = [], [], []
    for k in range(T):
        for (i, j) in links[k] & links[k + 1]:    # present at slot start and end
            U += [i, j]; V += [j, i]; K += [k, k]
    order = np.lexsort((np.array(U), np.array(K)))
    return ContactPlan(names, kinds, T,
                       np.array(U, dtype=np.int32)[order],
                       np.array(V, dtype=np.int32)[order],
                       np.array(K, dtype=np.int32)[order],
                       meta=dict(P=P, S=S, regime=regime, alt_km=alt_km, inc_deg=inc_deg, DT=dt))


if __name__ == "__main__":
    import time
    for reg in ("isl_full", "isl_intra", "no_isl"):
        t = time.time()
        cp = make_contact_plan(6, 11, 180, reg)
        print(reg, cp.N, "nodes", cp.n_contacts, "contacts", f"{time.time()-t:.2f}s",
              "per-slot", cp.n_contacts / cp.T)
