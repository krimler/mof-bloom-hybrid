# experiments/mpeg70_bloom_prefilter_learned.py
import os, time, argparse, json
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from skimage import measure

from src.datasets.mpeg7_loader import load_mpeg7_dataset
from src.mof.refs2d import sample_circle, eval_ref
from src.mof.tree_builder import build_learned_tree_from_dataset
from src.baselines.zernike_descriptor import zernike_descriptor

RESULTS_CSV = "tables/mpeg70_bloom_results_learned.csv"
ROUTES_TXT  = "tables/mpeg70_bloom_routes_learned.txt"

# ---------- utils ----------
def sample_support_on_grid(img, M=1024):
    contours = measure.find_contours(img, 0.5)
    if not contours: return None, None
    contour = max(contours, key=lambda x: x.shape[0])
    U, _ = sample_circle(M, deterministic=True)    # canonical grid
    h = (U @ contour.T).max(axis=1)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return h, U

def leaf_descriptor_ensemble(leaf, hK, U, topk=3):
    """
    Concatenate (OP, OI) for top-k refs by OP (z-scored later).
    """
    den = np.linalg.norm(hK) + 1e-12
    per=[]
    for R in leaf.refs:
        col = eval_ref(U, R).reshape(-1,1)
        c, *_ = np.linalg.lstsq(col, hK, rcond=None)
        hproj = col @ c
        op = np.linalg.norm(hproj)/den
        oi = np.linalg.norm(hK - hproj)/den
        per.append((op,oi))
    order = np.argsort([op for op,_ in per])[-int(topk):]
    vals=[]
    for idx in order:
        vals.extend([per[idx][0], per[idx][1]])
    return np.array(vals, dtype=np.float64)

def zscore(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd

def cosine_top1(vecs, y):
    S = cosine_similarity(vecs)
    correct=0
    for i in range(len(vecs)):
        S[i,i]=-1
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
def run(root_dir, M=1024, k_root=6, k_child=4, topk_leaf=3, rerank_k=60, save_routes=1):
    # Build learned tree (analytic + medoids) using all data as "train"
    tree = build_learned_tree_from_dataset(load_mpeg7_dataset, root_dir, M=M,
                                           k_root=k_root, k_child=k_child)

    # Load dataset again for evaluation (canonical U grid)
    data = load_mpeg7_dataset(root_dir)
    labels, H_list, U_list = [], [], []
    U_can, _ = sample_circle(M, deterministic=True)
    for lbl, img in data:
        h, U = sample_support_on_grid(img, M=M)
        if h is None: continue
        labels.append(lbl); H_list.append(h); U_list.append(U_can)  # ensure canonical U
    le = LabelEncoder(); y = le.fit_transform(labels)
    N = len(y)

    # Route & build leaf vectors
    mof_vecs=[]; routes=[]; times=[]
    for hK in H_list:
        t0 = time.time()
        leaf, best_ref, path = tree.route(hK, U_can, m0=1024, delta=0.01, max_m=8192)
        times.append(time.time()-t0); routes.append(path)
        v = leaf_descriptor_ensemble(leaf, hK, U_can, topk=topk_leaf)
        mof_vecs.append(v)
    mof_vecs = np.stack(mof_vecs, axis=0)

    # Normalize + cosine retrieval
    mof_vecs_n = zscore(mof_vecs)
    mof_leaf_acc = cosine_top1(mof_vecs_n, y)

    # Hybrid re-rank on top-k
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

    correct=0
    S = cosine_similarity(mof_vecs_n)
    for i in range(N):
        S[i,i] = -1
        idx = np.argsort(-S[i])[:rerank_k]
        zi = zern_get(i)
        best_j=i; best_s=-1
        for j in idx:
            zj = zern_get(j)
            denom = (np.linalg.norm(zi)*np.linalg.norm(zj) + 1e-12)
            s = float(np.dot(zi, zj) / denom) if denom>0 else -1
            if s > best_s:
                best_s, best_j = s, j
        correct += (y[best_j]==y[i])
    hybrid_acc = correct / N

    # Persist results
    os.makedirs("tables", exist_ok=True)
    row = {
        "tree": "learned+analytic",
        "M": int(M),
        "k_root": int(k_root),
        "k_child": int(k_child),
        "topk_leaf": int(topk_leaf),
        "rerank_k": int(rerank_k),
        "mof_leaf_acc": float(mof_leaf_acc),
        "hybrid_acc": float(hybrid_acc),
        "avg_route_time_s": float(np.mean(times)),
        "num_queries": int(N),
    }
    df = pd.DataFrame([row])
    out_csv = RESULTS_CSV
    if not os.path.isfile(out_csv) or os.path.getsize(out_csv)==0:
        df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, mode="a", header=False, index=False)

    if save_routes:
        dump_routes(routes, ROUTES_TXT)

    print("\n=== MOF–Bloom Prefilter (LEARNED tree) ===")
    for k,v in row.items():
        print(f"{k:>16s}: {v}")
    print(f"\nAppended to {out_csv}")
    if save_routes:
        print(f"Saved routes to {ROUTES_TXT}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MPEG7dataset")
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--k-root", type=int, default=6)
    ap.add_argument("--k-child", type=int, default=4)
    ap.add_argument("--topk-leaf", type=int, default=3)
    ap.add_argument("--rerank-k", type=int, default=60)
    ap.add_argument("--save-routes", type=int, default=1)
    args = ap.parse_args()
    run(args.root, M=args.M, k_root=args.k_root, k_child=args.k_child,
        topk_leaf=args.topk_leaf, rerank_k=args.rerank_k, save_routes=args.save_routes)

if __name__ == "__main__":
    cli()

