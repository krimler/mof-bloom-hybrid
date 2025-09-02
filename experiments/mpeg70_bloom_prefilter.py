import os, time, argparse, json
import numpy as np, pandas as pd
from skimage import measure
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from src.datasets.mpeg7_loader import load_mpeg7_dataset
from src.mof.refs2d import sample_circle, eval_ref, analytic_reference_library
from src.mof.bloom_tree import Node
from src.baselines.zernike_descriptor import zernike_descriptor

RESULTS_CSV = "tables/mpeg70_bloom_results.csv"
ROUTES_TXT  = "tables/mpeg70_bloom_routes.txt"

# ------------------------
# Utilities
# ------------------------
def contour_to_support(img, M=1024):
    """Extract largest contour and sample support h_K on M directions."""
    contours = measure.find_contours(img, 0.5)
    if not contours: 
        return None, None
    contour = max(contours, key=lambda x: x.shape[0])  # (N,2)
    U, _ = sample_circle(M)
    h = (U @ contour.T).max(axis=1)
    h[~np.isfinite(h)] = 0.0
    return h, U

def euclid_top1(vecs, y):
    correct=0
    for i in range(len(vecs)):
        d = np.linalg.norm(vecs - vecs[i], axis=1)
        d[i]=np.inf
        j = np.argmin(d)
        correct += (y[j]==y[i])
    return correct/len(vecs)

def build_default_tree():
    lib = analytic_reference_library()
    root = Node(lib["root"], name="root", tau=0.15)
    for idx, child_refs in enumerate(lib["children"]):
        root.children[idx] = Node(child_refs, name=f"child{idx}", tau=0.15)
    return root

def leaf_descriptor_top1(leaf, hK, U, M=1024):
    ops = []
    den = np.linalg.norm(hK[:M]) + 1e-12
    for R in leaf.refs:
        col = eval_ref(U[:M], R).reshape(-1,1)
        c, *_ = np.linalg.lstsq(col, hK[:M], rcond=None)
        hproj = col @ c
        op = np.linalg.norm(hproj)/den
        oi = np.linalg.norm(hK[:M] - hproj)/den
        ops.append((op, oi))
    a = int(np.argmax([o for o,_ in ops]))
    return np.array([ops[a][0], ops[a][1]])

def leaf_descriptor_ensemble(leaf, hK, U, M=1024, topk=2):
    vals = []
    den = np.linalg.norm(hK[:M]) + 1e-12
    per = []
    for R in leaf.refs:
        col = eval_ref(U[:M], R).reshape(-1,1)
        c, *_ = np.linalg.lstsq(col, hK[:M], rcond=None)
        hproj = col @ c
        op = np.linalg.norm(hproj)/den
        oi = np.linalg.norm(hK[:M] - hproj)/den
        per.append((op, oi))
    order = np.argsort([op for op,_ in per])[-topk:]
    for idx in order:
        vals.extend([per[idx][0], per[idx][1]])
    return np.array(vals)

def to_py_scalar(x):
    """Convert numpy scalars to plain Python for JSON."""
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def dump_routes(routes, path):
    with open(path, "w") as f:
        for p in routes[:20]:
            # Each p is a list of tuples (name, best_idx, op, rad, m) possibly with np types
            clean = []
            for t in p:
                clean.append(tuple(to_py_scalar(v) for v in t))
            f.write(json.dumps(clean) + "\n")

# ------------------------
# Main evaluation
# ------------------------
def run(root_dir, leaf_mode="top1", ensemble_k=2, m0=1024, max_m=8192, rerank_k=40):
    # Load dataset
    data = load_mpeg7_dataset(root_dir)
    labels, H_list, U_list = [], [], []
    for lbl, img in data:
        h, U = contour_to_support(img, M=1024)
        if h is None: 
            continue
        labels.append(lbl); H_list.append(h); U_list.append(U)
    le = LabelEncoder(); y = le.fit_transform(labels)
    N = len(y)

    # Build Bloom tree
    tree = build_default_tree()

    # Route queries and build MOF leaf descriptor
    mof_vecs = []; routes = []; times = []
    for hK, U in zip(H_list, U_list):
        t0 = time.time()
        leaf, best_ref, path = tree.route(hK, U, m0=m0, delta=0.01, max_m=max_m)
        times.append(time.time()-t0); routes.append(path)
        if leaf_mode == "ensemble":
            vec = leaf_descriptor_ensemble(leaf, hK, U, M=1024, topk=int(ensemble_k))
        else:
            vec = leaf_descriptor_top1(leaf, hK, U, M=1024)
        mof_vecs.append(vec)

    mof_vecs = np.stack(mof_vecs, axis=0)

    # Stage-1: MOF retrieval (Euclidean)
    mof_acc = euclid_top1(mof_vecs, y)

    # Stage-2: Hybrid re-rank: compute Zernike only on top-k candidates per query
    zern_cache = {}
    def zern_get(i):
        if i in zern_cache: return zern_cache[i]
        _, img = data[i]
        z = zernike_descriptor(img, radius=min(img.shape)//2, degree=8)
        zern_cache[i] = z
        return z

    correct = 0
    for i in range(N):
        d = np.linalg.norm(mof_vecs - mof_vecs[i], axis=1)
        idx = np.argsort(d)[:rerank_k+1]
        idx = idx[idx!=i][:rerank_k]
        zi = zern_get(i)
        best_j = i; best_s = -1
        for j in idx:
            zj = zern_get(j)
            s = float(np.dot(zi, zj) / (np.linalg.norm(zi)*np.linalg.norm(zj) + 1e-12))
            if s > best_s:
                best_s, best_j = s, j
        if y[best_j] == y[i]:
            correct += 1
    hybrid_acc = correct / N

    # Persist results (append mode)
    os.makedirs("tables", exist_ok=True)
    row = {
        "leaf_mode": leaf_mode,
        "ensemble_k": int(ensemble_k) if leaf_mode=="ensemble" else np.nan,
        "rerank_k": int(rerank_k),
        "mof_leaf_acc": float(mof_acc),
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

    # Save a few routes in JSON-safe form
    dump_routes(routes, ROUTES_TXT)

    # Print summary
    print("\n=== MOF–Bloom Prefilter (run summary) ===")
    for k,v in row.items():
        print(f"{k:>18s}: {v}")
    print(f"\nAppended to {RESULTS_CSV}")
    print(f"Saved first route traces to {ROUTES_TXT}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MPEG7dataset", help="MPEG-7 root dir")
    ap.add_argument("--leaf-mode", choices=["top1","ensemble"], default="top1",
                    help="Descriptor at leaf: top1 OP/OI or small ensemble")
    ap.add_argument("--ensemble-k", type=int, default=2, help="Top-k refs for ensemble (2 or 3)")
    ap.add_argument("--m0", type=int, default=1024, help="Initial directions per node")
    ap.add_argument("--max-m", type=int, default=8192, help="Max directions per node")
    ap.add_argument("--rerank-k", type=int, default=40, help="Top-k candidates for Zernike re-rank")
    args = ap.parse_args()
    run(args.root, leaf_mode=args.leaf_mode, ensemble_k=args.ensemble_k,
        m0=args.m0, max_m=args.max_m, rerank_k=args.rerank_k)

if __name__ == "__main__":
    cli()

