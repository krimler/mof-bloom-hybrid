# src/mof/refs2d.py
import numpy as np

# ------------------------
# Direction sampling (2D)
# ------------------------
def sample_circle(M, rng=None, deterministic=False):
    """
    Directions on S^1.
    - deterministic=True -> theta = linspace(0,2pi); otherwise uniform random
    Returns U=(M,2), theta=(M,).
    """
    if deterministic:
        theta = np.linspace(0.0, 2*np.pi, num=int(M), endpoint=False, dtype=np.float64)
    else:
        rng = np.random.default_rng(rng)
        theta = rng.uniform(0, 2*np.pi, size=int(M)).astype(np.float64)
    U = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return U, theta

# ------------------------
# Internal helpers
# ------------------------
def _safe_max_dot(U, V):
    try:
        if U is None or V is None:
            return None
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if U.ndim != 2 or V.ndim != 2 or U.shape[1] != 2 or V.shape[1] != 2:
            return None
        if U.shape[0] == 0 or V.shape[0] == 0:
            return np.zeros(U.shape[0], dtype=np.float64)
        if not np.isfinite(U).all() or not np.isfinite(V).all():
            return np.zeros(U.shape[0], dtype=np.float64)
        X = U @ V.T
        if not np.isfinite(X).all():
            return np.zeros(U.shape[0], dtype=np.float64)
        return X.max(axis=1)
    except Exception:
        return np.zeros(U.shape[0], dtype=np.float64)

def _angles_from_U(U):
    # map directions to angles in [0, 2pi)
    th = np.arctan2(U[:,1], U[:,0])
    th = np.where(th < 0, th + 2*np.pi, th)
    return th

# ------------------------
# Analytic 2D supports
# ------------------------
def support_polygon(U, k=4, radius=1.0, phi=0.0):
    if (k is None) or (k < 3) or (not np.isfinite(k)):
        return np.zeros(U.shape[0], dtype=np.float64)
    if radius is None or not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    phi = 0.0 if (phi is None or not np.isfinite(phi)) else float(phi)
    angles = phi + 2*np.pi*np.arange(int(k))/float(k)
    V = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    out = _safe_max_dot(U, V)
    return out if out is not None else np.zeros(U.shape[0], dtype=np.float64)

def support_ellipse(U, a=2.0, b=1.0, phi=0.0):
    a = 2.0 if (a is None or not np.isfinite(a) or a <= 0) else float(a)
    b = 1.0 if (b is None or not np.isfinite(b) or b <= 0) else float(b)
    phi = 0.0 if (phi is None or not np.isfinite(phi)) else float(phi)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c,-s],[s,c]], dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != 2 or not np.isfinite(U).all():
        return np.zeros(U.shape[0], dtype=np.float64)
    Urot = U @ R
    vals = (a*Urot[:,0])**2 + (b*Urot[:,1])**2
    vals = np.maximum(np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return np.sqrt(vals, dtype=np.float64)

def support_rod(U, L=3.0, phi=0.0):
    L = 3.0 if (L is None or not np.isfinite(L) or L <= 0) else float(L)
    phi = 0.0 if (phi is None or not np.isfinite(phi)) else float(phi)
    d = np.array([np.cos(phi), np.sin(phi)], dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2 or U.shape[1] != 2 or not np.isfinite(U).all():
        return np.zeros(U.shape[0], dtype=np.float64)
    proj = np.abs(U @ d)
    return L * np.nan_to_num(proj, nan=0.0, posinf=0.0, neginf=0.0)

def support_star(U, k=5, r_outer=1.0, r_inner=0.4, phi=0.0):
    if (k is None) or (k < 3) or (not np.isfinite(k)):
        return np.zeros(U.shape[0], dtype=np.float64)
    r_outer = 1.0 if (r_outer is None or not np.isfinite(r_outer) or r_outer <= 0) else float(r_outer)
    r_inner = 0.4 if (r_inner is None or not np.isfinite(r_inner) or r_inner <= 0) else float(r_inner)
    phi = 0.0 if (phi is None or not np.isfinite(phi)) else float(phi)
    pts = []
    for i in range(2*int(k)):
        ang = phi + i*np.pi/float(k)
        r = r_outer if (i % 2 == 0) else r_inner
        pts.append([r*np.cos(ang), r*np.sin(ang)])
    V = np.asarray(pts, dtype=np.float64)
    out = _safe_max_dot(U, V)
    return out if out is not None else np.zeros(U.shape[0], dtype=np.float64)

# ------------------------
# Medoid reference support (stored on canonical grid)
# ------------------------
def support_medoid(U, ref):
    """
    Evaluate a learned 'medoid' reference on arbitrary directions U:
    - ref['h']: (M0,) support samples on theta0 = linspace(0,2pi,M0)
    - We evaluate h at theta(U) by nearest-neighbor lookup (cheap, robust)
    """
    h0 = ref.get("h", None)
    if h0 is None:
        return np.zeros(U.shape[0], dtype=np.float64)
    h0 = np.asarray(h0, dtype=np.float64)
    M0 = h0.size
    th = _angles_from_U(U)
    # map theta to indices on canonical grid
    idx = np.floor(th * M0 / (2*np.pi)).astype(int) % M0
    vals = h0[idx]
    return np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

# ------------------------
# Reference evaluation (robust)
# ------------------------
def eval_ref(U, ref):
    kind = ref.get("kind", None)
    try:
        if kind == "kgon":
            return support_polygon(U, k=int(ref["k"]), radius=float(ref.get("r",1.0)),
                                   phi=float(ref.get("phi",0.0)))
        if kind == "ellipse":
            return support_ellipse(U, a=float(ref["a"]), b=float(ref["b"]),
                                   phi=float(ref.get("phi",0.0)))
        if kind == "rod":
            return support_rod(U, L=float(ref["L"]), phi=float(ref.get("phi",0.0)))
        if kind == "star":
            return support_star(U, k=int(ref.get("k",5)), r_outer=float(ref.get("ro",1.0)),
                                r_inner=float(ref.get("ri",0.4)), phi=float(ref.get("phi",0.0)))
        if kind == "medoid":
            return support_medoid(U, ref)
    except Exception:
        pass
    # fallback
    return np.zeros(U.shape[0], dtype=np.float64)

# ------------------------
# Analytic reference library (richer)
# ------------------------
def analytic_reference_library():
    """
    Small, diverse analytic references for silhouettes.
    Returns dict: {'root': [...], 'children': [[...], ...]}
    """
    root = [
        {"kind":"kgon","k":4},
        {"kind":"kgon","k":6},
        {"kind":"ellipse","a":2.0,"b":1.0,"phi":0.0},
        {"kind":"ellipse","a":3.0,"b":1.0,"phi":0.0},
        {"kind":"rod","L":4.0,"phi":0.0},
        {"kind":"star","k":5,"ro":1.0,"ri":0.45,"phi":0.0},
    ]
    children = [
        # under square
        [
            {"kind":"kgon","k":3},{"kind":"kgon","k":8},
            {"kind":"ellipse","a":1.5,"b":1.0,"phi":0.0},
            {"kind":"star","k":5,"ro":1.0,"ri":0.45,"phi":np.deg2rad(18)}
        ],
        # under hexagon
        [
            {"kind":"kgon","k":5},{"kind":"kgon","k":7},
            {"kind":"ellipse","a":1.5,"b":1.0,"phi":0.0},
            {"kind":"star","k":5,"ro":1.0,"ri":0.45,"phi":np.deg2rad(36)}
        ],
        # under ellipse 2:1
        [
            {"kind":"ellipse","a":2.0,"b":1.0,"phi":np.deg2rad(30)},
            {"kind":"ellipse","a":2.0,"b":1.0,"phi":np.deg2rad(60)},
            {"kind":"rod","L":3.0,"phi":np.deg2rad(30)},
            {"kind":"kgon","k":8}
        ],
        # under ellipse 3:1
        [
            {"kind":"ellipse","a":3.0,"b":1.0,"phi":np.deg2rad(30)},
            {"kind":"ellipse","a":3.0,"b":1.0,"phi":np.deg2rad(60)},
            {"kind":"rod","L":5.0,"phi":np.deg2rad(30)},
            {"kind":"kgon","k":8}
        ],
        # under rod
        [
            {"kind":"rod","L":4.0,"phi":np.deg2rad(30)},
            {"kind":"rod","L":5.0,"phi":np.deg2rad(60)},
            {"kind":"ellipse","a":2.5,"b":1.0,"phi":np.deg2rad(30)},
            {"kind":"kgon","k":3}
        ],
        # under star
        [
            {"kind":"star","k":5,"ro":1.0,"ri":0.40,"phi":np.deg2rad(18)},
            {"kind":"kgon","k":8},
            {"kind":"ellipse","a":1.5,"b":1.0,"phi":0.0}
        ],
    ]
    return {"root": root, "children": children}

