# experiments/mpeg70_bloom_prefilter_learned_split.py
import os, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from skimage import measure

from src.datasets.mpeg7_loader import load_mpeg7_dataset
from src.mof.refs2d import sample_circle, eval_ref
from src.mof.tree_builder import build_learned_tree_from_supports
from src.baselines.zernike_descriptor import zernike_descriptor

SPLIT_JSON = "data/splits/mpeg7_split_80_20.json"
OUT_CSV    = "tables/mpeg70_bloom_results_learned_split.csv"
ROUTES_TXT = "tables/mpeg70_bloom_routes_learned_split.txt"

# ---------- utilities ----------
def sample_support_on_grid(img, U):
    contours = measure.find_contours(img, 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda x: x.shape[0])
    h = (U @ contour.T).max(axis=1)
    return np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

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

# ---------- leaf descriptor ----------
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

# ---------- main ----------
def run(root_dir, rerank_k=60, M=1024, topk_leaf=3, save_routes=1):
    os.makedirs("tables", exist_ok=True)

    # 0) Load saved split
    if not os.path.isfile(SPLIT_JSON):
        raise FileNotFoundError(f"Missing split file {SPLIT_JSON}. Run experiments.mpeg70_make_split first.")
    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)

    data = load_mpeg7_dataset(root_dir)
    train_idx = split["train_idx"]
    test_idx  = split["test_idx"]

    # Canonical directions
    U_can, _ = sample_circle(M, deterministic=True)

    # 1) Train supports for learned medoids
    H_train=[]
    for i in train_idx:
        _, img = data[i]
        h = sample_support_on_grid(img, U_can)
        if h is not None:
            H_train.append(h)

    # 2) Build learned tree from train supports
    #    (k_root/k_child pulled from split JSON if present, else defaults)
    k_root  = int(split.get("k_root", 6))
    k_child = int(split.get("k_child", 4))
    tree = build_learned_tree_from_supports(H_train, U_can, k_root=k_root, k_child=k_child)

    # 3) Evaluate on test
    labels_test=[]; H_test=[]
    for i in test_idx:
        lbl, img = data[i]
        h = sample_support_on_grid(img, U_can)
        if h is None: continue
        labels_test.append(lbl); H_test.append(h)
    le = LabelEncoder(); y = le.fit_transform(labels_test)
    N = len(y)

    # Route & build leaf vectors
    mof_vecs=[]; routes=[]; times=[]
    for hK in H_test:
        t0=time.time()
        leaf, best_ref, path = tree.route(hK, U_can, m0=1024, delta=0.01, max_m=8192)
        times.append(time.time()-t0); routes.append(path)
        v = leaf_descriptor_ensemble(leaf, hK, U_can, topk=topk_leaf)
        mof_vecs.append(v)
    mof_vecs = np.stack(mof_vecs, axis=0)
    mof_vecs_n = zscore(mof_vecs)

    # Stage-1 leaf retrieval (cosine)
    mof_leaf_acc = cosine_top1(mof_vecs_n, y)

    # Stage-2 hybrid rerank (top-k) on test
    S = cosine_similarity(mof_vecs_n)
    correct = 0
    for i in range(N):
        S[i,i] = -1
        idx = np.argsort(-S[i])[:rerank_k]
        # zernike for i and candidates
        _, img_i = data[test_idx[i]]
        zi = zernike_descriptor(img_i, radius=min(img_i.shape)//2, degree=8)
        best_j=i; best_s=-1
        for j in idx:
            _, img_j = data[test_idx[j]]
            zj = zernike_descriptor(img_j, radius=min(img_j.shape)//2, degree=8)
            denom = (np.linalg.norm(zi)*np.linalg.norm(zj) + 1e-12)
            s = float(np.dot(zi, zj) / denom) if denom>0 else -1
            if s > best_s:
                best_s, best_j = s, j
        correct += (y[best_j]==y[i])
    hybrid_acc = correct / N

    # Zernike-full baseline on test
    zern = []
    t0 = time.time()
    for i in range(N):
        _, img = data[test_idx[i]]
        zern.append(zernike_descriptor(img, radius=min(img.shape)//2, degree=8))
    z_time = time.time() - t0
    zern = np.stack(zern, axis=0)
    S_full = cosine_similarity(zern)
    corr_full=0
    for i in range(N):
        S_full[i,i]=-1
        j = np.argmax(S_full[i])
        corr_full += (y[j]==y[i])
    zern_full_acc = corr_full / N
    zern_time_per_query = z_time / N

    # Persist
    row = {
        "tree": "learned+analytic",
        "split": f"{int((1-split['test_size'])*100)}/{int(split['test_size']*100)}",
        "M": int(M),
        "k_root": int(k_root),
        "k_child": int(k_child),
        "topk_leaf": int(topk_leaf),
        "rerank_k": int(rerank_k),
        "mof_leaf_acc": float(mof_leaf_acc),
        "hybrid_acc": float(hybrid_acc),
        "zernike_full_acc": float(zern_full_acc),
        "avg_route_time_s": float(np.mean(times)),
        "zern_full_time_s_per_query": float(zern_time_per_query),
        "num_test": int(N),
    }
    df = pd.DataFrame([row])
    if not os.path.isfile(OUT_CSV) or os.path.getsize(OUT_CSV)==0:
        df.to_csv(OUT_CSV, index=False)
    else:
        df.to_csv(OUT_CSV, mode="a", header=False, index=False)

    if save_routes:
        dump_routes(routes, ROUTES_TXT)

    # Print summary
    print("\n=== MOF–Bloom Prefilter (LEARNED tree, split) ===")
    for k,v in row.items():
        print(f"{k:>26s}: {v}")
    print(f"\nAppended to {OUT_CSV}")
    if save_routes:
        print(f"Saved routes to {ROUTES_TXT}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MPEG7dataset")
    ap.add_argument("--rerank-k", type=int, default=60)
    ap.add_argument("--M", type=int, default=1024)
    ap.add_argument("--topk-leaf", type=int, default=3)
    ap.add_argument("--save-routes", type=int, default=1)
    args = ap.parse_args()
    run(args.root, rerank_k=args.rerank_k, M=args.M, topk_leaf=args.topk_leaf, save_routes=args.save_routes)

if __name__ == "__main__":
    cli()

