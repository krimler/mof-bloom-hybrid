import os, pandas as pd, matplotlib.pyplot as plt

BLOOM_CSV = "tables/mpeg70_bloom_results_learned_split.csv"
BASE_CSV  = "tables/mpeg70_baselines_split.csv"
FIG       = "figures/mpeg70_split_combined_bars.pdf"

def main():
    if not os.path.isfile(BLOOM_CSV):
        raise FileNotFoundError(f"Missing {BLOOM_CSV}")
    if not os.path.isfile(BASE_CSV):
        raise FileNotFoundError(f"Missing {BASE_CSV}")

    db = pd.read_csv(BLOOM_CSV).iloc[-1]
    ds = pd.read_csv(BASE_CSV).iloc[-1]

    leaf   = float(db["mof_leaf_acc"])
    hybrid = float(db["hybrid_acc"])
    zfull  = float(db["zernike_full_acc"])
    z_ms   = float(db["zern_full_time_s_per_query"])*1000.0 if "zern_full_time_s_per_query" in db else None
    avg_ms = float(db["avg_route_time_s"])*1000.0 if "avg_route_time_s" in db else None

    # Baselines (test split)
    zernike_acc = float(ds["zernike_acc"])
    efd_acc     = float(ds["efd_simple_acc"])
    hu_acc      = float(ds["hu_acc"])
    split_str   = str(db.get("split","test"))

    labels = ["MOF-Leaf (6D)", f"Hybrid (top-{int(db['rerank_k'])})", "Zernike-full", "EFD-simple", "Hu moments"]
    vals   = [leaf, hybrid, zfull, efd_acc, hu_acc]

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8,4))
    bars = plt.bar(labels, vals)
    plt.ylim(0,1); plt.ylabel("Retrieval top-1 (test)")
    plt.title(f"MPEG-7 (split {split_str}): Bloom (learned) vs baselines")
    for b,v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2.0, v+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    footer=[]
    if avg_ms is not None: footer.append(f"Bloom route: {avg_ms:.2f} ms/query")
    if z_ms  is not None: footer.append(f"Zernike-full: {z_ms:.2f} ms/query")
    if footer:
        plt.gcf().text(0.5, -0.02, "  |  ".join(footer), ha="center", fontsize=9)

    plt.tight_layout(); plt.savefig(FIG); plt.close()
    print(f"Saved {FIG}")

if __name__ == "__main__":
    main()

