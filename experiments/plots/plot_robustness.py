"""
Plot robustness results from tables/robustness.csv
Generates line plots for smoothing, jitter, and partial drop experiments.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def run():
    df = pd.read_csv("tables/robustness.csv")
    os.makedirs("figures", exist_ok=True)

    # Smoothing
    df_smooth = df[df["type"]=="smoothing"]
    plt.figure()
    plt.plot(df_smooth["rho"], df_smooth["OP"], marker="o", label="OP")
    plt.plot(df_smooth["rho"], df_smooth["OI"], marker="s", label="OI")
    plt.xlabel("rho (smoothing)")
    plt.ylabel("value")
    plt.title("Minkowski smoothing")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/robustness_smoothing.pdf")

    # Jitter
    df_jitter = df[df["type"]=="jitter"]
    plt.figure()
    plt.plot(df_jitter["sigma"], df_jitter["OP"], marker="o", label="OP")
    plt.plot(df_jitter["sigma"], df_jitter["OI"], marker="s", label="OI")
    plt.xlabel("sigma (jitter)")
    plt.ylabel("value")
    plt.title("Jitter robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/robustness_jitter.pdf")

    # Partial views
    df_partial = df[df["type"]=="partial"]
    plt.figure()
    plt.plot(df_partial["frac_drop"], df_partial["OP"], marker="o", label="OP")
    plt.plot(df_partial["frac_drop"], df_partial["OI"], marker="s", label="OI")
    plt.xlabel("fraction dropped")
    plt.ylabel("value")
    plt.title("Partial direction robustness")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/robustness_partial.pdf")

    print("Saved robustness plots to figures/")

if __name__=="__main__":
    run()

