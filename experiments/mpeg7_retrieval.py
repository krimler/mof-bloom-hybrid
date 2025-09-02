"""
MPEG-7 retrieval and clustering experiment.
- Loads silhouettes from flat directory (e.g. apple-1.gif, watch-19.gif)
- Computes MOF(OP,OI), Fourier, and Zernike descriptors
- Evaluates retrieval (top-1) and clustering purity
- Saves CSV and bar plot (MOF vs Fourier vs Zernike)
- Also saves a grid of example silhouettes
"""

import os, argparse, random
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from skimage import measure
from scipy.spatial import ConvexHull

from src.datasets.mpeg7_loader import load_mpeg7_dataset
from src.mof.support import fibonacci_sphere, support_polytope
from src.mof.projector_dict import build_dictionary, project_onto_dictionary
from src.mof.opoi import op_oi
from src.baselines.fourier_descriptor import fourier_descriptor
from src.baselines.zernike_descriptor import zernike_descriptor


def convex_hull_2d(contour):
    hull = ConvexHull(contour)
    return contour[hull.vertices]


def contour_to_vertices(img):
    contours = measure.find_contours(img, 0.5)
    if not contours:
        return None
    return max(contours, key=lambda x: x.shape[0])


def evaluate_retrieval(features, labels):
    sim = cosine_similarity(features)
    correct, total = 0, len(labels)
    for i in range(total):
        sim[i, i] = -1
        j = np.argmax(sim[i])
        if labels[j] == labels[i]:
            correct += 1
    return correct / total


def evaluate_clustering(features, ytrue, k):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)
    purity = sum([np.max(np.bincount(ytrue[km.labels_ == c]))
                  for c in range(k)]) / len(ytrue)
    return purity


def run(root="data/MPEG7dataset"):
    # Load dataset
    data = load_mpeg7_dataset(root)
    labels_all = [lbl for lbl, _ in data]
    print(f"Loaded {len(data)} images from {len(set(labels_all))} classes.")

    # MOF setup
    M, d = 500, 4
    U = fibonacci_sphere(M)
    V_ref = np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],
                      [-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]])
    rots = np.array([np.eye(3) for _ in range(d)])
    H = build_dictionary(U, V_ref, rots)

    mof_features, fourier_feats, zernike_feats, labels = [], [], [], []

    for label, img in data:
        contour = contour_to_vertices(img)
        if contour is None:
            continue

        # MOF
        V2d = convex_hull_2d(contour)
        V = np.column_stack([V2d, np.zeros(len(V2d))])
        y = support_polytope(V, U)
        phi, _ = project_onto_dictionary(y, H)
        op, oi, _ = op_oi(y, phi)
        mof_features.append([op, oi])
        labels.append(label)

        # Fourier
        fd = fourier_descriptor(contour, K=10)
        fourier_feats.append(fd)

        # Zernike
        try:
            zd = zernike_descriptor(img, radius=min(img.shape)//2, degree=8)
        except Exception:
            zd = np.zeros(10)
        zernike_feats.append(zd)

    # Convert to arrays
    mof_features = np.array(mof_features)
    fourier_feats = np.array(fourier_feats)
    zernike_feats = np.array(zernike_feats)
    labels = np.array(labels)

    # Encode labels → integers
    le = LabelEncoder()
    ytrue = le.fit_transform(labels)
    k = len(set(labels))

    # Evaluate
    results = {}
    for name, feats in [("MOF", mof_features),
                        ("Fourier", fourier_feats),
                        ("Zernike", zernike_feats)]:
        retrieval = evaluate_retrieval(feats, labels)
        purity = evaluate_clustering(feats, ytrue, k)
        results[name] = {"retrieval_top1": retrieval, "clustering_purity": purity}
        print(name, results[name])

    # Save CSV
    rows = []
    for method, vals in results.items():
        rows.append({"method": method,
                     "retrieval_top1": vals["retrieval_top1"],
                     "clustering_purity": vals["clustering_purity"]})
    os.makedirs("tables", exist_ok=True)
    pd.DataFrame(rows).to_csv("tables/mpeg7_results.csv", index=False)

    # Bar plot
    os.makedirs("figures", exist_ok=True)
    df = pd.DataFrame(rows)
    x = np.arange(len(df["method"]))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - w/2, df["retrieval_top1"], w, label="Retrieval top-1")
    ax.bar(x + w/2, df["clustering_purity"], w, label="Clustering purity")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("MPEG-7: Retrieval & Clustering")
    ax.legend()
    fig.tight_layout()
    plt.savefig("figures/mpeg7_bar.pdf")
    plt.close()
    print("Saved bar plot to figures/mpeg7_bar.pdf")

    # Example silhouettes grid
    classes = random.sample(list(set(labels)), 5)
    fig, axs = plt.subplots(len(classes), 3, figsize=(6, 8))
    for i, cls in enumerate(classes):
        imgs = [img for (lbl, img) in data if lbl == cls]
        n = min(3, len(imgs))
        samples = random.sample(imgs, n)
        for j in range(3):
            axs[i, j].axis("off")
            if j < n:
                axs[i, j].imshow(samples[j], cmap="gray")
            if j == 1:
                axs[i, j].set_title(cls)
    plt.tight_layout()
    plt.savefig("figures/mpeg7_examples.pdf")
    plt.close()
    print("Saved example silhouettes to figures/mpeg7_examples.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/MPEG7dataset",
                        help="Path to MPEG7 dataset root folder")
    args = parser.parse_args()
    run(root=args.root)

