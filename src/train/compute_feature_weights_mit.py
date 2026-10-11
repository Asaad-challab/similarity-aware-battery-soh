import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.lstm import LSTMSOH

# ============================
# CONFIG (MUST MATCH TRAINING)
# ============================
FEATURES = [
    "t_chg", "t_V20", "t_V40", "t_V60", "t_V80",
    "Q_chg", "E_chg", "t_CV", "T_mean", "T_delta"
]

MIN_SOH = 70.0
MAX_SOH = 102.0


# ============================
# DATASET (ABSOLUTE SOH)
# ============================
class AbsSOHDataset(Dataset):
    def __init__(self, files, seq_len, feat_mean, feat_std):
        self.samples = []
        self.seq_len = seq_len

        for path in files:
            df = pd.read_csv(path)

            if "cycle" not in df.columns or "SOH" not in df.columns:
                continue

            df = df[(df["SOH"] >= MIN_SOH) & (df["SOH"] <= MAX_SOH)]
            df = df.sort_values("cycle").reset_index(drop=True)

            if len(df) < seq_len:
                continue

            X = df[FEATURES].values.astype(np.float32)
            X = (X - feat_mean) / feat_std
            y = df["SOH"].values.astype(np.float32)

            for t in range(seq_len - 1, len(df)):
                self.samples.append((
                    torch.tensor(X[t - seq_len + 1:t + 1]),
                    torch.tensor(y[t])
                ))

        if len(self.samples) == 0:
            raise RuntimeError("No validation samples created")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================
# MAIN
# ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit_ckpt", required=True)
    ap.add_argument("--mit_cfg", required=True)
    ap.add_argument("--mit_split", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # ----------------------------
    # Load MIT config (normalization)
    # ----------------------------
    with open(args.mit_cfg) as f:
        cfg = json.load(f)

    feat_mean = np.array(cfg["feat_mean"], dtype=np.float32)
    feat_std = np.maximum(np.array(cfg["feat_std"], dtype=np.float32), 1e-8)

    seq_len = int(cfg.get("seq_len", 20))

    # ----------------------------
    # Load MIT split.json
    # ----------------------------
    with open(args.mit_split) as f:
        split = json.load(f)

    val_files = split["val_files"]

    print(f"[INFO] Validation cells: {len(val_files)}")

    # ----------------------------
    # Dataset + loader
    # ----------------------------
    ds = AbsSOHDataset(
        files=val_files,
        seq_len=seq_len,
        feat_mean=feat_mean,
        feat_std=feat_std
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # ----------------------------
    # Load model
    # ----------------------------
    model = LSTMSOH(
        input_dim=len(FEATURES),
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    model.load_state_dict(torch.load(args.mit_ckpt, map_location=device))
    model.eval()

    # ----------------------------
    # Gradient-based feature importance
    # ----------------------------
    feat_importance = torch.zeros(len(FEATURES), device=device)
    count = 0

    loss_fn = nn.MSELoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x.requires_grad_(True)

        pred = model(x)
        loss = loss_fn(pred, y)

        model.zero_grad()
        loss.backward()

        # Mean absolute gradient per feature
        grads = x.grad.abs().mean(dim=(0, 1))  # (features,)
        feat_importance += grads
        count += 1

    feat_importance /= max(count, 1)

    # Normalize to sum = 1
    weights = feat_importance.detach().cpu().numpy()
    weights = weights / weights.sum()

    out = {
        "features": FEATURES,
        "weights": weights.tolist(),
        "method": "MIT gradient-based sensitivity",
        "normalization": "feature-only (SOH unnormalized)",
        "source_model": args.mit_ckpt
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("[RESULT] Feature weights:")
    for f, w in zip(FEATURES, weights):
        print(f"  {f:10s} : {w:.4f}")

    print(f"[INFO] Saved to {args.out_json}")


if __name__ == "__main__":
    main()
