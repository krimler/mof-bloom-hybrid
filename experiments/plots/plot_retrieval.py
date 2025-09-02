"""
Plot retrieval and clustering results from tables/retrieval.csv
Makes a simple bar chart comparing metrics.
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def run():
    df = pd.read_csv("tables/retrieval.csv")
    os.makedirs("figures", exist_ok=True)
    vals = df.iloc[0].to_dict()
    print("Results:", vals)

    plt.figure()
    plt.bar(["Retrieval top-1","Clustering purity"], [vals["retrieval_top1"], vals["clustering_purity"]])
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Retrieval & Clustering Accuracy")
    plt.tight_layout()
    plt.savefig("figures/retrieval_bar.pdf")

    print("Saved retrieval plot to figures/")

if __name__=="__main__":
    run()

