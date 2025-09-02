# experiments/mpeg70_bloom_prefilter_norm.py
import os, time, argparse, json
import numpy as np, pandas as pd
from skimage import measure
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from src.datasets.mpeg7_loader import load_mpeg7_dataset
from src.mof.refs2d import sample_circle, eval_ref, analytic_reference_library
from src.mof.bloom_tree import Node
from src.baselines.zernike_descriptor import zernike_descriptor

RESULTS_CSV = "tables/mpeg70_bloom_results_norm.csv"
ROUTES_TXT  = "tables/mpeg70_bloom_routes_norm.txt"

# ---------- utils ----------
def contour_to_support(img, M=1024):
    contours = measure.find_contours(img, 0.5)
    if not contours: return None, None
    contour = max(contours, key=lambda x: x.shape[0])
    U, _ = sample_circle(M)
    h = (U @ contour.T).max(axis=1)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return h, U

def build_default_tree():
    lib = analytic_reference_library()
    root = Node(lib["root"], name="root", tau=0.15)
    for idx, refs in enumerate(lib["children"]):
        root.children[idx] = Node(refs, name=f"child{idx}", tau=0.15)
    return root

def leaf_descriptor_top1(leaf, hK, U, M=1024):
    den = np.linalg.norm(hK[:M]) + 1e-12
    ops=[]
    for R in leaf.refs:
        col = eval_ref(U[:M], R).reshape(-1,1)
        c, *_ = np.linalg.lstsq(col, hK[:M], rcond=None)
        hproj = col @ c
        op = np.linalg.norm(hproj)/den
        oi = np.linalg.norm(hK[:M]-hproj)/den
        ops.append((op,oi))
    a = int(np.argmax([o for o,_ in ops]))
    return np.array([ops[a][0], ops[a][1]])

def leaf_descriptor_ensemble(leaf, hK, U, M=1024, topk=2):
    den = np.linalg.norm(hK[:M]) + 1e-12
    per=[]
    for R in leaf.refs:
        col = eval_ref(U[:M], R).reshape(-1,1)
        c, *_ = np.linalg.lstsq(col, hK[:M], rcond=None)
        hproj = col @ c
        op = np.linalg.norm(hproj)/den
        oi = np.linalg.norm(hK[:M]-hproj)/den
        per.append((op,oi))
    order = np.argsort([op for op,_ in per])[-topk:]
    vals=[]
    for idx in order:
        vals.extend([per[idx][0], per[idx][1]])
    return np.array(vals)

def zscore(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd

def euclid_top1(vecs, y):
    correct=0
    for i in range(len(vecs)):
        d = np.linalg.norm(vecs - vecs[i], axis=1)
        d[i]=np.inf
        j = np.argmin(d)
        correct += (y[j]==y[i])
    return correct/len(vecs)

def cosine_top1(vecs, y):
    S = cosine_similarity(vecs)
    correct=0
    for i in range(len(vecs)):
        S[i,i] = -1
        j = np.argmax(S[i])
        correct += (y[j]==y[i])
    return correct/len(vecs)

def dump_routes(routes, path, max_lines=20):
    def to_py(x):
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.integer,)):  return int(x)
        return x
    with open(path, "w") as f:
        for p in routes[:max_lines]:
            clean = [tuple(to_py(v) for v in t) for t in p]
            f.write(json.dumps(clean) + "\n")

# ---------- main ----------
def run(root_dir, leaf_mode="ensemble", ensemble_k=3, m0=1024, max_m=8192,
        rerank_k=60, normalize="zscore", distance="euclid", save_routes=1):

    # load
    data = load_mpeg7_dataset(root_dir)
    labels, H_list, U_list = [], [], []
    for lbl, img in data:
        h, U = contour_to_support(img, M=1024)
        if h is None: continue
        labels.append(lbl); H_list.append(h); U_list.append(U)
    le = LabelEncoder(); y = le.fit_transform(labels)
    N = len(y)

    # tree
    tree = build_default_tree()

    # route + build leaf vectors
    mof_vecs=[]; routes=[]; times=[]
    for hK, U in zip(H_list, U_list):
        t0=time.time()
        leaf, best_ref, path = tree.route(hK, U, m0=m0, delta=0.01, max_m=max_m)
        times.append(time.time()-t0); routes.append(path)
        if leaf_mode=="top1":
            v = leaf_descriptor_top1(leaf, hK, U, M=1024)
        else:
            v = leaf_descriptor_ensemble(leaf, hK, U, M=1024, topk=int(ensemble_k))
        mof_vecs.append(v)

    mof_vecs = np.stack(mof_vecs, axis=0)

    # normalize descriptor (optional)
    if normalize == "zscore":
        mof_vecs_use = zscore(mof_vecs)
    else:
        mof_vecs_use = mof_vecs

    # Stage-1 leaf retrieval
    if distance == "cosine":
        leaf_acc = cosine_top1(mof_vecs_use, y)
    else:
        leaf_acc = euclid_top1(mof_vecs_use, y)

    # Stage-2 hybrid: top-k candidates by the SAME distance
    correct = 0
    zern_cache={}
    def zern_get(i):
        if i in zern_cache: return zern_cache[i]
        _, img = data[i]
        try:
            z = zernike_descriptor(img, radius=min(img.shape)//2, degree=8)
        except Exception:
            z = np.zeros(10, dtype=np.float64)
        zern_cache[i] = z
        return z

    for i in range(N):
        if distance == "cosine":
            sims = cosine_similarity(mof_vecs_use[i:i+1], mof_vecs_use).ravel()
            sims[i] = -1
            idx = np.argsort(-sims)[:rerank_k]
        else:
            d = np.linalg.norm(mof_vecs_use - mof_vecs_use[i], axis=1)
            d[i]=np.inf
            idx = np.argsort(d)[:rerank_k]
        zi = zern_get(i)
        best_j, best_s = i, -1
        for j in idx:
            zj = zern_get(j)
            denom = (np.linalg.norm(zi)*np.linalg.norm(zj) + 1e-12)
            s = float(np.dot(zi, zj) / denom) if denom>0 else -1
            if s > best_s:
                best_s, best_j = s, j
        correct += (y[best_j]==y[i])
    hybrid_acc = correct / N

    # persist
    os.makedirs("tables", exist_ok=True)
    row = {
        "leaf_mode": leaf_mode,
        "ensemble_k": int(ensemble_k) if leaf_mode=="ensemble" else np.nan,
        "normalize": normalize,
        "distance": distance,
        "rerank_k": int(rerank_k),
        "mof_leaf_acc": float(leaf_acc),
        "hybrid_acc": float(hybrid_acc),
        "avg_route_time_s": float(np.mean(times)),
        "num_queries": int(N),
        "m0": int(m0),
        "max_m": int(max_m),
    }
    df = pd.DataFrame([row])
    if not os.path.isfile(RESULTS_CSV) or os.path.getsize(RESULTS_CSV)==0:
        df.to_csv(RESULTS_CSV, index=False)
    else:
        df.to_csv(RESULTS_CSV, mode="a", header=False, index=False)

    if save_routes:
        dump_routes(routes, ROUTES_TXT)

    print("\n=== MOF–Bloom Prefilter (normalized) ===")
    for k,v in row.items():
        print(f"{k:>18s}: {v}")
    print(f"\nAppended to {RESULTS_CSV}")
    if save_routes:
        print(f"Saved route traces to {ROUTES_TXT}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MPEG7dataset")
    ap.add_argument("--leaf-mode", choices=["top1","ensemble"], default="ensemble")
    ap.add_argument("--ensemble-k", type=int, default=3)
    ap.add_argument("--m0", type=int, default=1024)
    ap.add_argument("--max-m", type=int, default=8192)
    ap.add_argument("--rerank-k", type=int, default=60)
    ap.add_argument("--normalize", choices=["none","zscore"], default="zscore")
    ap.add_argument("--distance", choices=["euclid","cosine"], default="euclid")
    ap.add_argument("--save-routes", type=int, default=1)
    args = ap.parse_args()
    run(args.root, leaf_mode=args.leaf_mode, ensemble_k=args.ensemble_k,
        m0=args.m0, max_m=args.max_m, rerank_k=args.rerank_k,
        normalize=args.normalize, distance=args.distance, save_routes=args.save_routes)

if __name__ == "__main__":
    cli()

