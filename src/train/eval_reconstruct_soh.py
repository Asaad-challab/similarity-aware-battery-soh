import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.lstm import LSTMSOH

# ===============================
# Configuration
# ===============================
FEATURES = [
    "t_chg","t_V20","t_V40","t_V60","t_V80",
    "Q_chg","E_chg","t_CV","T_mean","T_delta"
]

MIN_SOH = 70.0
MAX_SOH = 102.0
SEQ_LEN = 20
PLOT_CELLS = 3


# ===============================
# Normalization (TRAIN ONLY)
# ===============================
def compute_norm_stats(train_files):
    X_all = []
    for f in train_files:
        df = pd.read_csv(f)
        df = df[(df["SOH"] >= MIN_SOH) & (df["SOH"] <= MAX_SOH)]
        X_all.append(df[FEATURES].values)

    X = np.vstack(X_all)
    return X.mean(axis=0), X.std(axis=0) + 1e-8


# ===============================
# Reconstruct SOH for ONE cell
# ===============================
def reconstruct_cell(df, model, feat_mean, feat_std):
    df = df[(df["SOH"] >= MIN_SOH) & (df["SOH"] <= MAX_SOH)]
    df = df.sort_values("cycle").reset_index(drop=True)

    N = len(df)
    if N < SEQ_LEN:
        return None

    X = df[FEATURES].values
    X = (X - feat_mean) / feat_std
    soh_true = df["SOH"].values

    model.eval()
    soh_pred = []

    with torch.no_grad():
        for t in range(SEQ_LEN - 1, N):
            x_seq = torch.tensor(
                X[t-(SEQ_LEN-1):t+1],
                dtype=torch.float32
            ).unsqueeze(0)

            delta = model(x_seq).item()
            soh_hat = soh_true[t-(SEQ_LEN-1)] + delta
            soh_pred.append(soh_hat)

    soh_true_aligned = soh_true[SEQ_LEN-1:]
    soh_pred = np.array(soh_pred)

    return soh_true_aligned, soh_pred


# ===============================
# Main
# ===============================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    split = json.load(open(os.path.join(args.run_dir, "split.json")))
    train_files = split["train_files"]
    test_files  = split["test_files"]

    feat_mean, feat_std = compute_norm_stats(train_files)

    device = torch.device("cpu")

    model = LSTMSOH(
        input_dim=10,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(args.run_dir, "best_model.pt"),
            map_location=device
        )
    )

    mae_list  = []
    rmse_list = []
    r2_list   = []

    plot_dir = os.path.join(args.run_dir, "reconstruction_plots")
    os.makedirs(plot_dir, exist_ok=True)

    for idx, csv_path in enumerate(test_files):
        df = pd.read_csv(csv_path)
        result = reconstruct_cell(df, model, feat_mean, feat_std)

        if result is None:
            continue

        soh_true, soh_pred = result

        mae  = np.mean(np.abs(soh_true - soh_pred))
        rmse = np.sqrt(np.mean((soh_true - soh_pred) ** 2))

        ss_res = np.sum((soh_true - soh_pred) ** 2)
        ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        if idx < PLOT_CELLS:
            plt.figure()
            plt.plot(soh_true, label="True SOH")
            plt.plot(soh_pred, "--", label="Reconstructed SOH")
            plt.xlabel("Cycle")
            plt.ylabel("SOH")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"cell_{idx}_soh.png"))
            plt.close()

    metrics = {
        "absolute_SOH_MAE_mean":  float(np.mean(mae_list)),
        "absolute_SOH_MAE_std":   float(np.std(mae_list)),
        "absolute_SOH_RMSE_mean": float(np.mean(rmse_list)),
        "absolute_SOH_RMSE_std":  float(np.std(rmse_list)),
        "R2_mean":               float(np.mean(r2_list)),
        "R2_std":                float(np.std(r2_list)),
        "num_test_cells":        len(mae_list)
    }

    with open(os.path.join(args.run_dir, "reconstruction_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nABSOLUTE SOH RECONSTRUCTION METRICS (MIT, 100%)")
    print(metrics)


if __name__ == "__main__":
    main()
