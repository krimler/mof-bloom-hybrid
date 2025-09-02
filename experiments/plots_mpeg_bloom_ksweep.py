import os, subprocess, sys
import pandas as pd
import matplotlib.pyplot as plt

RAW   = "tables/mpeg70_bloom_results.csv"
CLEAN = "tables/mpeg70_bloom_results_clean.csv"
FIG   = "figures/mpeg70_bloom_ksweep.pdf"

def ensure_clean():
    if os.path.isfile(CLEAN):
        return
    if not os.path.isfile(RAW):
        raise FileNotFoundError(f"Missing {RAW}. Run the prefilter first.")
    # normalize mixed schemas (3-col legacy + 9-col new) into CLEAN
    subprocess.check_call([sys.executable, "-m", "experiments.fix_mpeg70_results"])

def main():
    os.makedirs("figures", exist_ok=True)
    ensure_clean()
    df = pd.read_csv(CLEAN)
    if df.empty:
        raise ValueError(f"{CLEAN} is empty.")
    # keep rows with rerank_k present
    dff = df.dropna(subset=["rerank_k"]).copy()
    dff["rerank_k"] = dff["rerank_k"].astype(int)
    dff = dff.sort_values("rerank_k")
    # get last row for route time (ms)
    last = df.iloc[-1]
    avg_ms = None
    if "avg_route_time_s" in last and pd.notna(last["avg_route_time_s"]):
        avg_ms = float(last["avg_route_time_s"]) * 1000.0

    plt.figure(figsize=(6,4))
    plt.plot(dff["rerank_k"], dff["hybrid_acc"], marker="o", label="Hybrid (MOF→Zernike)")
    # optional: if you ran leaf-mode=leaf rows, you could also plot mof_leaf_acc vs k (flat)
    plt.xlabel("Top-k for Zernike re-rank")
    plt.ylabel("Retrieval top-1 accuracy")
    plt.title("70-class MPEG-7: Hybrid accuracy vs top-k")
    plt.ylim(0,1)
    plt.grid(alpha=0.3)
    if avg_ms is not None:
        plt.gcf().text(0.5, -0.02, f"Avg route time (last run): {avg_ms:.2f} ms/query",
                       ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG)
    plt.close()
    print(f"Saved {FIG}\n\nLast few rows:\n", dff.tail(5).to_string(index=False))

if __name__ == "__main__":
    main()

