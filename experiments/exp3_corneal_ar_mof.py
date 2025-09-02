#!/usr/bin/env python3
"""
Experiment 3: Corneal Endothelium Morphometry with AR-MOF (Hexagon Reference)

This script computes OP–OI values for corneal endothelial cells using
analytic-reference MOF (AR-MOF) with the hexagon as the natural reference.
It then produces three figures:
  1. OP–OI scatter (per cell, colored by polygon sides)
  2. Distribution of OP by polygon class
  3. Correlation between image-level mean OP and hexagonality %
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.mof.opoi import op_oi

# -----------------------------
# Helpers
# -----------------------------

def clean_percent_column(series: pd.Series) -> pd.Series:
    """Convert percentage strings like '89.5%' to floats."""
    return pd.to_numeric(series.astype(str).str.replace("%", ""), errors="coerce")

def compute_cell_opoi(row, hex_shape_index=3.72):
    """Compute OP/OI for one cell row relative to hexagon reference."""
    sides = row["Cell - Number of Sides"]
    shape_index = row["Cell - Shape Index"]

    # Observed vector: [sides, shape_index]
    y = np.array([sides, shape_index], dtype=float)
    # Reference vector: [6, hex_shape_index]
    phi = np.array([6, hex_shape_index], dtype=float)

    op, oi, _ = op_oi(y, phi)
    return op, oi

# -----------------------------
# Main
# -----------------------------

def main(args):
    # Load dataset
    df = pd.read_csv(args.csv)

    # Clean % columns we care about
    if "Image - Mean Hexagonal Cell Regularity" in df.columns:
        df["Image - Mean Hexagonal Cell Regularity"] = clean_percent_column(
            df["Image - Mean Hexagonal Cell Regularity"]
        )

    if "Image - Cell Area CV" in df.columns:
        df["Image - Cell Area CV"] = clean_percent_column(df["Image - Cell Area CV"])

    if "Cell - Hexagonal Cell Regularity" in df.columns:
        df["Cell - Hexagonal Cell Regularity"] = clean_percent_column(
            df["Cell - Hexagonal Cell Regularity"]
        )

    # Compute OP, OI per cell
    df[["OP", "OI"]] = df.apply(compute_cell_opoi, axis=1, result_type="expand")
    df = df.dropna(subset=["OP", "OI"])

    # Prepare output directory
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)

    # Plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 14,
        "figure.figsize": (8, 6),
        "savefig.format": "pdf"
    })

    # -----------------------------
    # 1. OP–OI Scatter (per cell)
    # -----------------------------
    plt.figure()
    sns.scatterplot(
        data=df.sample(min(10000, len(df)), random_state=42),
        x="OP", y="OI",
        hue="Cell - Number of Sides",
        palette="tab10",
        alpha=0.6
    )
    plt.title("OP–OI Scatter (Hexagon Reference)")
    plt.xlabel("Order Proportion (OP)")
    plt.ylabel("Order Irregularity (OI)")
    plt.legend(title="Polygon sides", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "exp3_scatter.pdf")
    plt.close()

    # -----------------------------
    # 2. Distribution of OP by Polygon Class
    # -----------------------------
    plt.figure()
    sns.violinplot(
        data=df,
        x="Cell - Number of Sides",
        y="OP",
        palette="muted",
        inner="box"
    )
    plt.title("Distribution of OP by Polygon Side Count")
    plt.xlabel("Number of Sides")
    plt.ylabel("Order Proportion (OP)")
    plt.tight_layout()
    plt.savefig(out_dir / "exp3_op_distribution.pdf")
    plt.close()

    # -----------------------------
    # 3. Correlation: Mean OP vs. Hexagonality %
    # -----------------------------
    image_op = df.groupby("Image Number")["OP"].mean().reset_index(name="Mean_OP")

    if "Image - Mean Hexagonal Cell Regularity" in df.columns:
        hex_reg = df.groupby("Image Number")["Image - Mean Hexagonal Cell Regularity"].first().reset_index()
        merged = pd.merge(image_op, hex_reg, on="Image Number")
        merged = merged.dropna(subset=["Image - Mean Hexagonal Cell Regularity", "Mean_OP"])

        plt.figure()
        sns.scatterplot(
            data=merged,
            x="Image - Mean Hexagonal Cell Regularity",
            y="Mean_OP",
            alpha=0.7
        )
        sns.regplot(
            data=merged,
            x="Image - Mean Hexagonal Cell Regularity",
            y="Mean_OP",
            scatter=False,
            color="red"
        )
        plt.title("Mean OP vs. Hexagonal Cell Regularity (%)")
        plt.xlabel("Hexagonality % (Image-level)")
        plt.ylabel("Mean OP (Image-level)")
        plt.tight_layout()
        plt.savefig(out_dir / "exp3_op_vs_hexagonality.pdf")
        plt.close()
    else:
        print("⚠️ Column 'Image - Mean Hexagonal Cell Regularity' not found. Skipping correlation plot.")

    print(f"✅ Figures saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Corneal AR-MOF Analysis")
    parser.add_argument("--csv", type=str, required=True, help="Path to Human_Corneal_Endothelium.csv")
    parser.add_argument("--outdir", type=str, default="exp3_figures", help="Output directory for figures")
    args = parser.parse_args()
    main(args)

# python experiments/exp3_corneal_ar_mof.py --csv data/Human_Corneal_Endothelium.csv

