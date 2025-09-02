.ONESHELL:

setup:
\tpython -m venv .venv
\t. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .

exp1:
\tPYTHONPATH=. python experiments/nd_demo.py

exp2:
\tPYTHONPATH=. python experiments/mpeg70_bloom_prefilter_norm.py --root data/MPEG7dataset --top-k 60
\tPYTHONPATH=. python experiments/plots_mpeg_bloom.py --results tables/mpeg70_bloom_results_norm.csv

exp3:
\tPYTHONPATH=. python experiments/exp3_corneal_ar_mof.py --csv data/Human_Corneal_Endothelium.csv --outdir figures

all: exp1 exp2 exp3

