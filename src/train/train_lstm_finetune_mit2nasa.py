import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.lstm import LSTMSOH

# ============================================================
# CONFIG
# ============================================================
FEATURES = [
    "t_chg","t_V20","t_V40","t_V60","t_V80",
    "Q_chg","E_chg","t_CV","T_mean","T_delta"
]

MIN_SOH = 70.0
MAX_SOH = 102.0


def log(msg):
    print(f"[INFO] {msg}")


# ============================================================
# IO / DATA
# ============================================================
def load_cell_df(csv_path):
    df = pd.read_csv(csv_path)

    # Basic sanity
    if "cycle" not in df.columns or "SOH" not in df.columns:
        raise KeyError(f"{csv_path} missing required columns: cycle/SOH")

    # Filter + sort
    df = df[(df["SOH"] >= MIN_SOH) & (df["SOH"] <= MAX_SOH)].copy()
    df = df.sort_values("cycle").reset_index(drop=True)

    # Feature existence check
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} missing FEATURES: {missing}")

    return df


def resolve_split_files(split, data_dir):
    """
    Accepts multiple split formats. Your NASA LOCO files are:
    {'train_files','val_files','test_files'}.
    """
    keys = list(split.keys())
    log(f"Split keys detected: {keys}")

    if "train_files" in split:
        train_list = split.get("train_files", [])
        val_list = split.get("val_files", [])
        test_list = split.get("test_files", [])
        log("Using train_files + val_files for fine-tuning (LOCO correct)")
    elif "train" in split:
        train_list = split["train"]
        val_list = split.get("val", [])
        test_list = split.get("test", [])
        log("Using train/val/test keys")
    else:
        raise KeyError(f"Unknown split format: {split.keys()}")

    def abs_paths(lst):
        out = []
        for f in lst:
            out.append(f if os.path.isabs(f) else os.path.join(data_dir, f))
        return out

    return abs_paths(train_list), abs_paths(val_list), abs_paths(test_list)


def load_mit_norm_from_ckpt(mit_ckpt_path):
    """
    CRITICAL: fine-tuning must use the SAME normalization as MIT pretraining.
    We assume MIT training wrote config.json next to best_model.pt.
    """
    mit_dir = os.path.dirname(mit_ckpt_path)
    cfg_path = os.path.join(mit_dir, "config.json")

    log(f"Loading MIT normalization from: {cfg_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"MIT config.json not found next to checkpoint.\n"
            f"Expected: {cfg_path}\n"
            f"Your mit_ckpt: {mit_ckpt_path}"
        )

    with open(cfg_path) as f:
        cfg = json.load(f)

    if "feat_mean" not in cfg or "feat_std" not in cfg:
        raise KeyError(
            f"MIT config.json missing feat_mean/feat_std.\n"
            f"Keys found: {list(cfg.keys())}"
        )

    mean = np.array(cfg["feat_mean"], dtype=np.float32)
    std = np.array(cfg["feat_std"], dtype=np.float32)

    if mean.shape[0] != len(FEATURES) or std.shape[0] != len(FEATURES):
        raise ValueError(
            f"MIT mean/std length mismatch.\n"
            f"mean len={len(mean)} std len={len(std)} expected={len(FEATURES)}"
        )

    log(f"MIT mean (first 3): {mean[:3]}")
    log(f"MIT std  (first 3): {std[:3]}")

    # Safety: avoid divide-by-zero
    std = np.maximum(std, 1e-8)

    return mean, std


# ============================================================
# DATASET (ΔSOH windows)
# ============================================================
class DeltaSOHDataset(Dataset):
    """
    Inputs: normalized features (using MIT mean/std)
    Targets: ΔSOH = SOH(t) - SOH(t-seq_len+1)
    """
    def __init__(self, files, seq_len, mean, std, max_windows=None, tag=""):
        self.samples = []
        self.seq_len = seq_len

        log(f"Building ΔSOH windows {tag} (seq_len={seq_len})")

        for path in files:
            log(f"  Loading {os.path.basename(path)}")
            df = load_cell_df(path)

            if len(df) < seq_len:
                log("   ↳ skipped (too short)")
                continue

            X = df[FEATURES].values.astype(np.float32)
            X = (X - mean) / std
            soh = df["SOH"].values.astype(np.float32)

            # windows end at t = seq_len-1 .. N-1
            for t in range(seq_len - 1, len(df)):
                x_seq = X[t - (seq_len - 1): t + 1]
                y_delta = soh[t] - soh[t - (seq_len - 1)]
                self.samples.append((
                    torch.tensor(x_seq, dtype=torch.float32),
                    torch.tensor(y_delta, dtype=torch.float32),
                ))

        if len(self.samples) == 0:
            raise RuntimeError("No ΔSOH samples created. Check seq_len / data quality.")

        if max_windows is not None:
            max_windows = max(1, int(max_windows))
            log(f"Reducing dataset to {max_windows} windows (ft_ratio)")
            self.samples = self.samples[:max_windows]

        log(f"Final {tag} samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# DIAGNOSTICS
# ============================================================
def tensor_finite_report(x, name="tensor"):
    finite_mask = torch.isfinite(x)
    all_finite = bool(finite_mask.all().item())
    if all_finite:
        return True, f"{name}: all finite"
    # Find a few bad indices
    bad = (~finite_mask).nonzero(as_tuple=False)
    bad = bad[:5].cpu().numpy().tolist()
    return False, f"{name}: NOT finite. First bad indices: {bad}"


def diagnose_batch(x, y, pred):
    okx, msgx = tensor_finite_report(x, "x")
    oky, msgy = tensor_finite_report(y, "y")
    okp, msgp = tensor_finite_report(pred, "pred")

    log("[DIAGNOSTIC SNAPSHOT]")
    log(msgx)
    log(msgy)
    log(msgp)

    # min/max (guarded)
    def safe_minmax(t):
        if not torch.isfinite(t).any().item():
            return ("nan", "nan")
        tf = t[torch.isfinite(t)]
        return (float(tf.min().item()), float(tf.max().item()))

    ymn, ymx = safe_minmax(y)
    pmn, pmx = safe_minmax(pred)

    log(f"y min/max    : {ymn}, {ymx}")
    log(f"pred min/max : {pmn}, {pmx}")
    log("=================================")

    return okx and oky and okp


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split_file", required=True)
    ap.add_argument("--mit_ckpt", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ft_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    device = torch.device(args.device)
    log(f"Device: {device}")

    # ----------------------------
    # Load split
    # ----------------------------
    log(f"Loading split file: {args.split_file}")
    with open(args.split_file) as f:
        split = json.load(f)

    train_files, val_files, test_files = resolve_split_files(split, args.data_dir)

    # We fine-tune on TRAIN+VAL (common in LOCO FT; test stays untouched)
    ft_files = train_files + val_files

    for f in ft_files:
        log(f"Fine-tune source: {os.path.basename(f)}")
    for f in test_files:
        log(f"Held-out TEST cell: {os.path.basename(f)}")

    # ----------------------------
    # Load MIT normalization stats (CRITICAL)
    # ----------------------------
    mean, std = load_mit_norm_from_ckpt(args.mit_ckpt)

    # ----------------------------
    # Build dataset and subsample windows for ft_ratio
    # ----------------------------
    full_ds = DeltaSOHDataset(ft_files, args.seq_len, mean, std, tag="(full)")
    n_ft = max(1, int(len(full_ds) * args.ft_ratio))
    ft_ds = DeltaSOHDataset(ft_files, args.seq_len, mean, std, max_windows=n_ft, tag="(ft)")

    loader = DataLoader(ft_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # ----------------------------
    # Model
    # ----------------------------
    log(f"Loading MIT checkpoint weights: {args.mit_ckpt}")
    if not os.path.exists(args.mit_ckpt):
        raise FileNotFoundError(f"mit_ckpt not found: {args.mit_ckpt}")

    model = LSTMSOH(
        input_dim=len(FEATURES),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    model.load_state_dict(torch.load(args.mit_ckpt, map_location=device))

    # Freeze LSTM layers to keep representation stable
    log("Freezing LSTM layers (fine-tune head only)")
    trainable = 0
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if "lstm" in name.lower():
            p.requires_grad = False
        else:
            p.requires_grad = True
            trainable += p.numel()
    log(f"Trainable params: {trainable} / {total}")

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.MSELoss()

    # ----------------------------
    # Training
    # ----------------------------
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            # Print diagnostics on the very first batch of epoch 1
            if ep == 1 and step == 0:
                ok = diagnose_batch(x, y, pred)
                if not ok:
                    raise RuntimeError("Non-finite tensors detected BEFORE loss. Fix data/normalization.")

            loss = loss_fn(pred, y)

            if not torch.isfinite(loss).item():
                # extra debug snapshot
                log("NaN/Inf loss detected. Printing diagnostic snapshot again:")
                diagnose_batch(x, y, pred)
                raise RuntimeError("NaN loss detected — aborting fine-tuning.")

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            losses.append(float(loss.item()))

        ep_mse = float(np.mean(losses)) if len(losses) else float("inf")
        log(f"Epoch {ep:02d} | FT ΔSOH MSE = {ep_mse:.6f}")
        history.append({"epoch": ep, "ft_mse": ep_mse})

    # ----------------------------
    # Save
    # ----------------------------
    best_path = os.path.join(args.run_dir, "best_model.pt")
    torch.save(model.state_dict(), best_path)

    # Save FT config (includes MIT norm provenance)
    out_cfg = vars(args).copy()
    out_cfg["features"] = FEATURES
    out_cfg["mit_norm_from"] = os.path.join(os.path.dirname(args.mit_ckpt), "config.json")
    with open(os.path.join(args.run_dir, "config.json"), "w") as f:
        json.dump(out_cfg, f, indent=2)

    with open(os.path.join(args.run_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    log(f"Saved fine-tuned model: {best_path}")
    log("Fine-tuning complete.")


if __name__ == "__main__":
    main()
