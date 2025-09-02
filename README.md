# Mixed Order Functionals (MOF) with Bloom Prefilter and Biomedical Extension

This repository accompanies the IEEE Access manuscript *Mixed Order Functionals: Symmetry-Filtered Support Projections with Stability and PAC Guarantees*.

We provide the **theory, code, and experiments** for Mixed Order Functionals (MOF), including:
- Analytic-reference MOF (AR-MOF)
- MOF--Bloom Prefilter for fast retrieval
- Applications in synthetic relativity tests, MPEG-7 shape retrieval, and corneal endothelium morphometry

---

## 🚀 Highlights

- **Theory**: MOF provides two interpretable scalars (Order Proportion OP, Order Irregularity OI) with exact identity, stability, and PAC guarantees.
- **Synthetic tests (Exp. 1)**: OP--OI scatter reproduces known geometric families across dimensions.
- **Benchmark retrieval (Exp. 2)**: Hybrid MOF--Bloom + Zernike matches Zernike accuracy (0.592 top-1) with 6.8× speedup.
- **Biomedical plausibility (Exp. 3)**: AR-MOF (hexagon reference) aligns with standard morphometric indices in corneal endothelium, using only tabular (privacy-preserved) data.

---

## 📂 Repository Structure

```
src/                # MOF core modules and baselines
  mof/              # MOF core (opoi, refs, bloom, tree builder)
  baselines/        # Zernike, EFD, Hu, Fourier, Spherical Harmonics
  datasets/         # MPEG-7 and synthetic loaders

experiments/        # Scripts for each experiment and plotting
figures/            # Paper-ready figures (PDF)
tables/             # CSV/TXT result tables
data/               # External datasets (MPEG-7, corneal CSV)
README.md           # This file
requirements.txt    # Dependencies
LICENSE             # MIT License
```

---

## 🔧 Installation

```bash
git clone https://github.com/<your-username>/mof-bloom-hybrid.git
cd mof-bloom-hybrid

# Install dependencies
pip install -r requirements.txt

# Install in dev mode
pip install -e .
```

Python ≥ 3.9 recommended.

---

## 🧪 Reproducing Experiments

### Experiment 1: Synthetic Relativity

```bash
PYTHONPATH=. python experiments/nd_demo.py
```
Generates `figures/nd_scatter_combined.pdf`.

### Experiment 2: MPEG-7 Retrieval

```bash
# Prefilter and hybrid retrieval
PYTHONPATH=. python experiments/mpeg70_bloom_prefilter_norm.py --root data/MPEG7dataset --top-k 60

# Plot results
PYTHONPATH=. python experiments/plots_mpeg_bloom.py --results tables/mpeg70_bloom_results_norm.csv
```
Generates `figures/mpeg70_split_combined_bars.pdf`.

### Experiment 3: Corneal Endothelium Morphometry

```bash
PYTHONPATH=. python experiments/exp3_corneal_ar_mof.py --csv data/Human_Corneal_Endothelium.csv --outdir figures
```
Generates:
- `figures/exp3_scatter.pdf`
- `figures/exp3_op_distribution.pdf`
- `figures/exp3_op_vs_hexagonality.pdf`

Dataset: [Figshare link](https://auckland.figshare.com/articles/dataset/Measurements_of_ex_vivo_human_corneal_endothelium_using_Voronoi_segmentation_/5701087).

**Note:** dataset is privacy-preserved (tabular only, no raw images).

---

## 📊 Figures in Paper

- Exp. 1: `nd_scatter_combined.pdf`
- Exp. 2: `mpeg70_split_combined_bars.pdf`
- Exp. 3: `exp3_scatter.pdf`, `exp3_op_distribution.pdf`, `exp3_op_vs_hexagonality.pdf`

See `figures/README.md` for full details.

---

## 📜 License

Released under the MIT License. See `LICENSE` for details.

---

## 📖 Artifact Availability

All code, experiment scripts, and plotting utilities are released in this repository.  
Figures in the paper can be reproduced by running the scripts above with provided datasets and CSV tables.

For questions, contact: <your email>.
