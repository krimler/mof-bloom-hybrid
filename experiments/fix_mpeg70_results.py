# experiments/fix_mpeg70_results.py
import os, csv, pandas as pd

RAW = "tables/mpeg70_bloom_results.csv"
CLEAN = "tables/mpeg70_bloom_results_clean.csv"

# unified column order
NEW_COLS = [
    "leaf_mode", "ensemble_k", "rerank_k",
    "mof_leaf_acc", "hybrid_acc", "avg_route_time_s",
    "num_queries", "m0", "max_m"
]

def main():
    if not os.path.isfile(RAW):
        raise FileNotFoundError(f"Missing {RAW}. Run the prefilter first.")

    rows = []
    with open(RAW, "r", newline="") as f:
        reader = csv.reader(f)
        # try to detect the old header: first row has 3 names
        header = next(reader, None)
        old3 = ["mof_leaf_acc", "hybrid_acc", "avg_route_time_s"]
        use_old3 = header is not None and [h.strip() for h in header] == old3

        # if first line isn't 3-col header, treat it as data
        if not use_old3 and header:
            # guess whether it's a 9-col row (new) or 3-col row (old)
            vals = [v.strip() for v in header]
            if len(vals) == 3:
                # old data row without header? map to unified dict
                rows.append({
                    "leaf_mode": "leaf", "ensemble_k": "",
                    "rerank_k": "", "mof_leaf_acc": vals[0],
                    "hybrid_acc": vals[1], "avg_route_time_s": vals[2],
                    "num_queries": "", "m0": "", "max_m": ""
                })
            elif len(vals) == 9:
                rows.append({k: v for k, v in zip(NEW_COLS, vals)})

        for vals in reader:
            vals = [v.strip() for v in vals]
            if len(vals) == 0:  # skip blank
                continue
            if len(vals) == 3:
                # old format row
                rows.append({
                    "leaf_mode": "leaf", "ensemble_k": "",
                    "rerank_k": "", "mof_leaf_acc": vals[0],
                    "hybrid_acc": vals[1], "avg_route_time_s": vals[2],
                    "num_queries": "", "m0": "", "max_m": ""
                })
            elif len(vals) == 9:
                rows.append({k: v for k, v in zip(NEW_COLS, vals)})
            else:
                # unknown row length; skip
                continue

    # build DataFrame with unified columns
    df = pd.DataFrame(rows, columns=NEW_COLS)

    # coerce numeric columns
    num_cols = ["ensemble_k", "rerank_k", "mof_leaf_acc", "hybrid_acc",
                "avg_route_time_s", "num_queries", "m0", "max_m"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs("tables", exist_ok=True)
    df.to_csv(CLEAN, index=False)
    print(f"Normalized CSV written to {CLEAN}")
    print("Last few rows:\n", df.tail().to_string(index=False))

if __name__ == "__main__":
    main()

