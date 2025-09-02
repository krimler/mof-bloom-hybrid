import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
df = pd.read_csv("tables/nd_demo.csv")

# === 1. OP/OI summary for baseline shapes ===
print("\n=== OP/OI Summary (baseline shapes) ===")
print(
    df[df["shape"].isin(["cube", "xpoly", "ellip_3to1", "ellip_2to1"])]
    .pivot(index="shape", columns="n", values="OP")
)

# === 2. OP vs OI scatter for baseline families ===
baseline = df[df["shape"].isin(["cube", "xpoly", "ellip_3to1", "ellip_2to1"])]
for n in sorted(baseline["n"].unique()):
    sub = baseline[baseline["n"] == n]
    plt.scatter(sub["OP"], sub["OI"], label=sub["shape"].values, s=80)
    for i, row in sub.iterrows():
        plt.text(row["OP"] + 0.01, row["OI"], row["shape"], fontsize=8)
    plt.plot([0, 1], [1, 0], "k--", alpha=0.5)  # diagonal guide
    plt.xlabel("OP")
    plt.ylabel("OI")
    plt.title(f"OP vs OI scatter ({n}D baseline shapes)")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"figures/nd_scatter_{n}D.pdf")
    plt.clf()

# === 3. Robustness: smoothing ===
rob_smooth = df[df["shape"].str.contains("smooth")]
for n in sorted(rob_smooth["n"].unique()):
    sub = rob_smooth[rob_smooth["n"] == n].copy()
    sub["rho"] = sub["shape"].str.extract(r"smooth([\d\.]+)").astype(float)
    plt.plot(sub["rho"], sub["OP"], marker="o", label=f"{n}D-OP")
    plt.plot(sub["rho"], sub["OI"], marker="s", linestyle="--", label=f"{n}D-OI")
plt.xlabel("Smoothing ρ")
plt.ylabel("Value")
plt.title("OP/OI vs smoothing (ellipsoid 3:1)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/nd_robust_smoothing.pdf")
plt.clf()

# === 4. Robustness: jitter ===
rob_jitter = df[df["shape"].str.contains("jitter")]
for n in sorted(rob_jitter["n"].unique()):
    sub = rob_jitter[rob_jitter["n"] == n].copy()
    sub["sigma"] = sub["shape"].str.extract(r"jitter([\d\.]+)").astype(float)
    plt.plot(sub["sigma"], sub["OP"], marker="o", label=f"{n}D-OP")
    plt.plot(sub["sigma"], sub["OI"], marker="s", linestyle="--", label=f"{n}D-OI")
plt.xlabel("Jitter σ")
plt.ylabel("Value")
plt.title("OP/OI vs jitter (ellipsoid 3:1)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/nd_robust_jitter.pdf")
plt.clf()

# === 5. Efficiency scaling ===
eff = df[df["shape"].str.contains("time")]
plt.loglog(eff["n"], eff["OI"], marker="o")
plt.xlabel("Dimension n")
plt.ylabel("Projection time (s)")
plt.title("Projection time scaling (ellip 3:1)")
plt.tight_layout()
plt.savefig("figures/nd_efficiency.pdf")
plt.clf()

# === Combined OP–OI scatter across dimensions ===
baseline = df[df["shape"].isin(["cube", "xpoly", "ellip_3to1", "ellip_2to1"])].copy()

plt.figure(figsize=(6,6))
shapes = baseline["shape"].unique()
markers = {2:"o",3:"s",4:"^",6:"D",8:"v",10:"P",12:"X"}  # marker per dimension

for shape in shapes:
    sub = baseline[baseline["shape"]==shape]
    for _, row in sub.iterrows():
        plt.scatter(row["OP"], row["OI"], 
                    color={"cube":"C0","xpoly":"C1","ellip_3to1":"C2","ellip_2to1":"C3"}[shape],
                    marker=markers.get(row["n"],"o"),
                    s=80, alpha=0.8,
                    label=f"{shape}-{row['n']}D")

# Diagonal guide
plt.plot([0,1],[1,0],"k--",alpha=0.4)

# Make legend unique
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05,1), loc="upper left", fontsize=7)

plt.xlabel("OP")
plt.ylabel("OI")
plt.title("Combined OP–OI scatter (baseline shapes, 2D–12D)")
plt.xlim(0,1.05); plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig("figures/nd_scatter_combined.pdf")
plt.clf()

print("Saved all plots to figures/")

