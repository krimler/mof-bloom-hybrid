# experiments/plots_mpeg_bloom.py
import os, subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt

RAW   = "tables/mpeg70_bloom_results.csv"
CLEAN = "tables/mpeg70_bloom_results_clean.csv"
FIG   = "figures/mpeg70_bloom_bars.pdf"

def ensure_clean():
    if os.path.isfile(CLEAN):
        return
    if not os.path.isfile(RAW):
        raise FileNotFoundError(f"Missing {RAW}. Run the prefilter first.")
    # call normalizer
    print("Normalizing mixed CSV ->", CLEAN)
    subprocess.check_call([sys.executable, "-m", "experiments.fix_mpeg70_results"])

def main():
    ensure_clean()
    df = pd.read_csv(CLEAN)
    if df.empty:
        raise ValueError(f"{CLEAN} is empty.")
    row = df.iloc[-1]  # last run

    # extract values with fallbacks
    leaf_mode = row.get("leaf_mode", "leaf")
    ensemble_k = row.get("ensemble_k")
    rerank_k = row.get("rerank_k")

    mof_acc = float(row["mof_leaf_acc"])
    hybrid_acc = float(row["hybrid_acc"])
    avg_ms = float(row["avg_route_time_s"])*1000.0 if "avg_route_time_s" in row else None

    if pd.notna(ensemble_k) and leaf_mode=="ensemble":
        leaf_label = f"MOF-Leaf (ensemble, top-{int(ensemble_k)})"
    else:
        leaf_label = f"MOF-Leaf ({leaf_mode})"

    if pd.notna(rerank_k):
        hybrid_label = f"Hybrid (top-{int(rerank_k)} Zernike)"
    else:
        hybrid_label = "Hybrid (Zernike re-rank)"

    plt.figure(figsize=(6,4))
    bars = plt.bar([leaf_label, hybrid_label], [mof_acc, hybrid_acc])
    plt.ylim(0,1)
    plt.ylabel("Retrieval top-1 accuracy")
    plt.title("70-class MPEG-7: Bloom Prefilter & Hybrid Re-rank")

    # annotate
    for b,val in zip(bars, [mof_acc, hybrid_acc]):
        plt.text(b.get_x()+b.get_width()/2.0, val+0.02, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=10)

    if avg_ms is not None:
        plt.gcf().text(0.5, -0.02, f"Avg route time: {avg_ms:.2f} ms/query",
                       ha="center", fontsize=9)

    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG)
    plt.close()

    print(f"Saved {FIG}")
    print(df.tail(3).to_string(index=False))

if __name__ == "__main__":
    main()

