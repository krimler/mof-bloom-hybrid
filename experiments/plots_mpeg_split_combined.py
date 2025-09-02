import os
import pandas as pd
import matplotlib.pyplot as plt

LEARNED_CSV = "tables/mpeg70_bloom_results_learned_split.csv"
BASE_CSV    = "tables/mpeg70_baselines_split.csv"
OUT_FIG     = "figures/mpeg70_split_combined_bars.pdf"

def read_last_row(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    return df.iloc[-1], df

def main():
    os.makedirs("figures", exist_ok=True)

    # Read learned/split Bloom results and baselines on the SAME test split
    learned_row, learned_df = read_last_row(LEARNED_CSV)
    base_row, base_df       = read_last_row(BASE_CSV)

    # Extract values (with fallbacks)
    split       = learned_row.get("split", "train/test")
    rerank_k    = int(learned_row.get("rerank_k", 0)) or None
    leaf_acc    = float(learned_row["mof_leaf_acc"])
    hybrid_acc  = float(learned_row["hybrid_acc"])
    zfull_acc   = float(learned_row.get("zernike_full_acc", base_row.get("zernike_acc", 0.0)))

    efd_k       = int(base_row.get("efd_k", 20)) if not pd.isna(base_row.get("efd_k", 20)) else 20
    efd_acc     = float(base_row["efd_simple_acc"])
    hu_acc      = float(base_row["hu_acc"])

    avg_route_s = float(learned_row.get("avg_route_time_s", 0.0))
    z_ms_full   = float(learned_row.get("zern_full_time_s_per_query", 0.0)) * 1000.0 if "zern_full_time_s_per_query" in learned_row else None

    # Build labels/values
    leaf_label   = "MOF-Leaf (6D)"
    hybrid_label = f"Hybrid (top-{rerank_k})" if rerank_k is not None else "Hybrid (top-k)"
    zfull_label  = "Zernike-full"
    efd_label    = f"EFD (K={efd_k})"
    hu_label     = "Hu moments"

    labels = [leaf_label, hybrid_label, zfull_label, efd_label, hu_label]
    vals   = [leaf_acc,   hybrid_acc,  zfull_acc,   efd_acc,   hu_acc]

    # ---- Plot (with safe margins) ----
    # Larger fig + extra bottom margin to avoid cropping x labels
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=False)
    bars = ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"])

    ax.set_ylim(0, 1)
    ax.set_ylabel("Retrieval top-1 (test)", fontsize=11)
    ax.set_title(f"MPEG-7 (split {split}): Bloom (learned) vs. Baselines", fontsize=12)

    # Rotate tick labels and right-align so long labels fit
    ax.set_xticklabels(labels, rotation=22, ha="right")

    # Annotate bars with values
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2.0, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    # Footer: route time and Zernike-full per-query time (if available)
    footer_parts = []
    if avg_route_s > 0:
        footer_parts.append(f"Bloom route: {avg_route_s*1000:.2f} ms/query")
    if z_ms_full is not None and z_ms_full > 0:
        footer_parts.append(f"Zernike-full: {z_ms_full:.2f} ms/query")
    if footer_parts:
        fig.text(0.5, 0.02, "  |  ".join(footer_parts), ha="center", fontsize=9)

    # Give extra bottom room for rotated labels; also use bbox_inches='tight'
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT_FIG}")
    print("\nLast learned row:\n", learned_row.to_string())
    print("\nLast baselines row:\n", base_row.to_string())

if __name__ == "__main__":
    main()

