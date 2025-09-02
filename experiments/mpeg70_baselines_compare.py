# experiments/mpeg70_baselines_compare.py
import os, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from skimage import measure

from src.datasets.mpeg7_loader import load_mpeg7_dataset
from src.baselines.zernike_descriptor import zernike_descriptor
from src.baselines.efd_simple import efd_simple
from src.baselines.hu_moments import hu_descriptor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

OUT_CSV = "tables/mpeg70_baselines.csv"
OUT_FIG = "figures/mpeg70_baselines_bars.pdf"

def extract_contour(img):
    contours = measure.find_contours(img, 0.5)
    if not contours:
        return None
    return max(contours, key=lambda x: x.shape[0])

def top1_acc_cosine(X, y):
    S = cosine_similarity(X)
    correct=0
    for i in range(len(y)):
        S[i,i] = -1
        j = np.argmax(S[i])
        correct += (y[j]==y[i])
    return correct/len(y)

def top1_acc_euclid(X, y):
    correct=0
    for i in range(len(y)):
        d = np.linalg.norm(X - X[i], axis=1)
        d[i] = np.inf
        j = np.argmin(d)
        correct += (y[j]==y[i])
    return correct/len(y)

def run(root, efd_k=20):
    os.makedirs("tables", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    data = load_mpeg7_dataset(root)
    labels, Z, EFD, HU = [], [], [], []
    for lbl, img in data:
        cnt = extract_contour(img)
        if cnt is None:
            continue
        labels.append(lbl)
        # Zernike
        try:
            z = zernike_descriptor(img, radius=min(img.shape)//2, degree=8)
        except Exception:
            z = np.zeros(10, dtype=np.float64)
        Z.append(z)
        # EFD simple
        try:
            e = efd_simple(cnt, K=efd_k)
        except Exception:
            e = np.zeros(efd_k, dtype=np.float64)
        EFD.append(e)
        # Hu moments
        try:
            h = hu_descriptor(img)
        except Exception:
            h = np.zeros(7, dtype=np.float64)
        HU.append(h)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    Z = np.stack(Z, axis=0)
    EFD = np.stack(EFD, axis=0)
    HU = np.stack(HU, axis=0)

    # retrieval accuracy
    z_acc   = top1_acc_cosine(Z, y)     # cosine works well on Zernike
    efd_acc = top1_acc_cosine(EFD, y)   # cosine on normalized magnitudes
    hu_acc  = top1_acc_euclid(HU, y)    # Euclidean on 7 Hu moments

    df = pd.DataFrame([{
        "zernike_acc": z_acc,
        "efd_simple_acc": efd_acc,
        "hu_acc": hu_acc,
        "efd_k": int(efd_k),
        "num": int(len(y))
    }])
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))

    # plot bars
    plt.figure(figsize=(6,4))
    bars = plt.bar(["Zernike (deg=8)", f"EFD (K={efd_k})", "Hu moments"], [z_acc, efd_acc, hu_acc])
    plt.ylim(0,1); plt.ylabel("Retrieval top-1 (full 70 classes)")
    plt.title("MPEG-7 Baselines")
    for b,v in zip(bars, [z_acc, efd_acc, hu_acc]):
        plt.text(b.get_x()+b.get_width()/2.0, v+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_FIG); plt.close()
    print(f"Saved {OUT_FIG}")

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MPEG7dataset")
    ap.add_argument("--efd-k", type=int, default=20)
    args = ap.parse_args()
    run(args.root, efd_k=args.efd_k)

if __name__ == "__main__":
    cli()

