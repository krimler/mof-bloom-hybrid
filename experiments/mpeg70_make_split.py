# experiments/mpeg70_make_split.py
import os, json, argparse, numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from src.datasets.mpeg7_loader import load_mpeg7_dataset

OUT_DIR  = "data/splits"
OUT_JSON = os.path.join(OUT_DIR, "mpeg7_split_80_20.json")

def main(root, test_size=0.2, seed=42, min_per_class=2):
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_mpeg7_dataset(root)
    labels = [lbl for lbl, _ in data]
    n_total = len(labels)
    print(f"Loaded {n_total} images from {len(set(labels))} classes")

    # Count per class and filter singletons
    cnt = Counter(labels)
    keep_mask = np.array([cnt[lbl] >= min_per_class for lbl in labels], dtype=bool)
    kept_idx  = np.where(keep_mask)[0]
    dropped   = np.where(~keep_mask)[0]

    if dropped.size > 0:
        dropped_classes = sorted({labels[i] for i in dropped})
        print(f"[WARN] Dropping {dropped.size} images from {len(dropped_classes)} class(es) with < {min_per_class} samples:")
        print("       " + ", ".join(dropped_classes[:10]) + (" ..." if len(dropped_classes) > 10 else ""))

    labels_kept = [labels[i] for i in kept_idx]
    le = LabelEncoder()
    y = le.fit_transform(labels_kept)
    idx = np.arange(len(kept_idx))

    if len(set(y)) < 2:
        raise RuntimeError("Not enough classes after filtering to build a split.")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_loc, test_loc = next(sss.split(idx, y))
    train_idx = kept_idx[train_loc].tolist()
    test_idx  = kept_idx[test_loc].tolist()

    payload = {
        "root": root,
        "test_size": float(test_size),
        "seed": int(seed),
        "min_per_class": int(min_per_class),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "num_total_seen": n_total,
        "num_kept": int(kept_idx.size),
        "num_dropped": int(dropped.size),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "num_classes_kept": int(len(set(labels_kept)))
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f)
    print(f"Saved split to {OUT_JSON}")
    print(f"num_total_seen={n_total}  kept={kept_idx.size}  dropped={dropped.size}")
    print(f"classes_kept={payload['num_classes_kept']}  train={len(train_idx)}  test={len(test_idx)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MPEG7dataset")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=2)
    args = ap.parse_args()
    main(args.root, test_size=args.test_size, seed=args.seed, min_per_class=args.min_per_class)

