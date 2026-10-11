import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.lstm import LSTMSOH

# ---------------------------------
# Configuration
# ---------------------------------
FEATURES = [
    "t_chg","t_V20","t_V40","t_V60","t_V80",
    "Q_chg","E_chg","t_CV","T_mean","T_delta"
]

MIN_SOH = 70.0
MAX_SOH = 102.0


# ---------------------------------
# Normalization (TRAIN ONLY)
# ---------------------------------
def compute_norm_stats(train_files):
    X_all = []

    for f in train_files:
        df = pd.read_csv(f)
        df = df[(df["SOH"] >= MIN_SOH) & (df["SOH"] <= MAX_SOH)]
        X_all.append(df[FEATURES].values)

    X = np.vstack(X_all)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    return mean, std


# ---------------------------------
# Dataset (ΔSOH target)
# ---------------------------------
class FileFilteredDataset(Dataset):
    def __init__(self, file_list, seq_len, feat_mean, feat_std):
        self.samples = []
        self.seq_len = seq_len
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        for csv_path in file_list:
            df = pd.read_csv(csv_path)

            df = df[(df["SOH"] >= MIN_SOH) & (df["SOH"] <= MAX_SOH)]
            df = df.sort_values("cycle").reset_index(drop=True)

            if len(df) < self.seq_len:
                continue

            X = df[FEATURES].values
            X = (X - self.feat_mean) / self.feat_std  # NORMALIZATION
            y = df["SOH"].values

            for i in range(len(df) - self.seq_len + 1):
                x_seq = torch.tensor(
                    X[i : i + self.seq_len],
                    dtype=torch.float32
                )

                delta_soh = y[i + self.seq_len - 1] - y[i]

                y_tgt = torch.tensor(
                    delta_soh,
                    dtype=torch.float32
                )

                self.samples.append((x_seq, y_tgt))

        if not self.samples:
            raise RuntimeError("No sequences created for this split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------
# Evaluation
# ---------------------------------
def evaluate(model, loader, device):
    model.eval()
    mae = 0.0
    mse = 0.0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            err = pred - y

            mae += err.abs().sum().item()
            mse += (err ** 2).sum().item()
            n += y.numel()

    return mae / n, np.sqrt(mse / n)


# ---------------------------------
# Training
# ---------------------------------
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    split_path = os.path.join(args.run_dir, "split.json")
    if not os.path.exists(split_path):
        raise RuntimeError(f"Missing {split_path}")

    split = json.load(open(split_path))

    # ---- normalization stats (TRAIN ONLY)
    feat_mean, feat_std = compute_norm_stats(split["train_files"])

    # ---- datasets
    train_ds = FileFilteredDataset(split["train_files"], args.seq_len, feat_mean, feat_std)
    val_ds   = FileFilteredDataset(split["val_files"],   args.seq_len, feat_mean, feat_std)
    test_ds  = FileFilteredDataset(split["test_files"],  args.seq_len, feat_mean, feat_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model
    model = LSTMSOH(
        input_dim=10,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_path = os.path.join(args.run_dir, "best_model.pt")

    # ---- training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            n += y.size(0)

        train_mse = total_loss / n
        val_mae, val_rmse = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_MSE={train_mse:.4f} | "
            f"val_MAE={val_mae:.4f} | "
            f"val_RMSE={val_rmse:.4f}"
        )

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)

    # ---- final test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate(model, test_loader, device)

    metrics = {
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_val_RMSE": best_val,
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
    }

    with open(os.path.join(args.run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTEST METRICS:", metrics)


if __name__ == "__main__":
    main()
