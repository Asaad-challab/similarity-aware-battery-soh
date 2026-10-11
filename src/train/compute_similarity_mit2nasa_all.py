#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import pandas as pd

MIT_DIR = os.environ.get("MIT_DATA_DIR", "data/MIT/merged")
NASA_DIR = os.environ.get("NASA_DATA_DIR", "data/NASA/merged")

WEIGHTS_JSON = "results/mit_feature_weights.json"
OUT_CSV = "src/train/results/similarity_report_mit2nasa_ALL.csv"

RATIOS = [0.10, 0.30, 0.50]
N_MIN = 25

FEATURE_COLS = [
    "t_chg","t_V20","t_V40","t_V60","t_V80","Q_chg","E_chg","t_CV","T_mean","T_delta"
]

# Load MIT weights
with open(WEIGHTS_JSON, "r") as f:
    raw = json.load(f)
weight_map = dict(zip(raw["features"], raw["weights"]))
WEIGHTS = np.array([weight_map[k] for k in FEATURE_COLS], np.float32)
WEIGHTS = WEIGHTS / WEIGHTS.sum()

# Utilities
def early_life_mean(df, ratio):
    n = len(df)
    n_obs = max(int(ratio*n), N_MIN)
    n_obs = min(n_obs, n)
    return df.iloc[:n_obs][FEATURE_COLS].mean().values

def compute_source_stats(mit_files, ratio):
    feats = []
    for f in mit_files:
        df = pd.read_csv(f).sort_values("cycle")
        feats.append(early_life_mean(df, ratio))
    feats = np.vstack(feats)
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0) + 1e-6
    d_w_all = np.sum(WEIGHTS * ((feats - mu) / sigma)**2, axis=1)
    tau = np.percentile(d_w_all, 75)
    return mu, sigma, tau

def compute_weighted(x, mu, sigma): return np.sum(WEIGHTS * ((x - mu)/sigma)**2)
def compute_l2(x, mu): return np.linalg.norm(x - mu)
def compute_cosine(x, mu): return 1 - (x@mu)/(np.linalg.norm(x)*np.linalg.norm(mu))

# Main
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    mit_files = sorted(glob.glob(os.path.join(MIT_DIR,"*.csv")))
    nasa_files = sorted(glob.glob(os.path.join(NASA_DIR,"B*.csv")))

    records=[]
    for ratio in RATIOS:
        mu,sigma,tau = compute_source_stats(mit_files, ratio)
        for f in nasa_files:
            cell = os.path.basename(f).replace(".csv","")
            df = pd.read_csv(f).sort_values("cycle")
            x  = early_life_mean(df, ratio)

            D_w  = compute_weighted(x, mu, sigma)
            D_l2 = compute_l2(x, mu)
            D_cos = compute_cosine(x, mu)

            records.append({
                "dataset":"NASA","cell":cell,"ratio":int(ratio*100),
                "D_w":float(D_w),"D_L2":float(D_l2),"D_cos":float(D_cos),
                "tau":float(tau),"similar":D_w <= tau
            })

    out = pd.DataFrame(records)
    out["rank"]= out.groupby("ratio")["D_w"].rank(method="min")
    out = out.sort_values(["ratio","D_w"])
    out.to_csv(OUT_CSV,index=False)

    print("\nSaved MIT→NASA ALL metrics CSV:")
    print(OUT_CSV)
    for r in [10,30,50]:
        print(f"\n=== {r}% NASA summary ===")
        print(out[out["ratio"]==r][["cell","D_w","D_L2","D_cos","rank","tau","similar"]].to_string(index=False))

if __name__=="__main__":
    main()
