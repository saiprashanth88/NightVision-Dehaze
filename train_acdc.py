import argparse
import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from acdc_dataset import ACDCPairedDataset, SyntheticDegradationDataset
from model import NightDehazeNet


CONDITIONS = ("fog", "night", "rain", "snow")


def ssim_loss(x, y, window=7):
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    sigma_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return 1.0 - score.clamp(0, 1).mean()


def edge_loss(x, y):
    """Sobel-gradient loss to discourage blurry road/vehicle edges."""
    gray_x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    gray_y = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx, gy = F.conv2d(gray_x, kx, padding=1), F.conv2d(gray_x, ky, padding=1)
    gx_t, gy_t = F.conv2d(gray_y, kx, padding=1), F.conv2d(gray_y, ky, padding=1)
    return F.l1_loss(gx, gx_t) + F.l1_loss(gy, gy_t)


def psnr_per_image(x, y):
    """Return one PSNR value per image, not one value for the whole batch."""
    mse = (x - y).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    values = 10.0 * torch.log10(1.0 / mse)
    return values


def ssim_per_image(x, y, window=7):
    """Return one SSIM value per image using the same local-window formula as ssim_loss."""
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, 1, pad)
    mu_y = F.avg_pool2d(y, window, 1, pad)
    sigma_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return score.clamp(0, 1).mean(dim=(1, 2, 3))


def condition_names(value, batch_size):
    if isinstance(value, (tuple, list)):
        return list(value)
    return [str(value)] * batch_size


def run_epoch(model, loader, optimizer, scaler, device, train=True, save_dir=None):
    model.train(train)
    total_loss = total_psnr = total_ssim = total_input_psnr = total_input_ssim = n = 0
    by_condition = defaultdict(lambda: {
        "loss": 0.0, "psnr": 0.0, "ssim": 0.0,
        "input_psnr": 0.0, "input_ssim": 0.0, "n": 0
    })
    saved = set()

    for x, y, condition, paths in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            out = model(x).clamp(0, 1)
            l1 = F.l1_loss(out, y)
            sl = ssim_loss(out, y)
            el = edge_loss(out, y)
            loss = l1 + 0.20 * sl + 0.05 * el

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        per_psnr = psnr_per_image(out.detach(), y)
        per_ssim = ssim_per_image(out.detach(), y)
        input_psnr = psnr_per_image(x, y)
        input_ssim = ssim_per_image(x, y)
        total_loss += loss.item() * bs
        total_psnr += float(per_psnr.sum().item())
        total_ssim += float(per_ssim.sum().item())
        total_input_psnr += float(input_psnr.sum().item())
        total_input_ssim += float(input_ssim.sum().item())
        n += bs

        names = condition_names(condition, bs)
        per_sample_loss = F.l1_loss(out.detach(), y, reduction="none").mean(dim=(1, 2, 3))
        for i, name in enumerate(names):
            stats = by_condition[name]
            stats["loss"] += float(per_sample_loss[i].item())
            stats["psnr"] += float(per_psnr[i].item())
            stats["ssim"] += float(per_ssim[i].item())
            stats["input_psnr"] += float(input_psnr[i].item())
            stats["input_ssim"] += float(input_ssim[i].item())
            stats["n"] += 1

            if save_dir and name not in saved:
                os.makedirs(save_dir, exist_ok=True)
                def to_bgr(t):
                    arr = t[i].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    return cv2.cvtColor((arr * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
                # One compact side-by-side example per validation category.
                montage = np.concatenate([to_bgr(x), to_bgr(out), to_bgr(y)], axis=1)
                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
                cv2.imwrite(os.path.join(save_dir, safe_name + ".jpg"), montage)
                saved.add(name)

    if n == 0:
        raise RuntimeError("DataLoader produced no batches.")
    summary = {k: {metric: value / stats["n"] for metric, value in stats.items()
                   if metric != "n"} for k, stats in by_condition.items()}
    overall = {
        "loss": total_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
        "input_psnr": total_input_psnr / n,
        "input_ssim": total_input_ssim / n,
    }
    return overall, summary


def main():
    p = argparse.ArgumentParser(
        description="Train NightVision-Dehaze with real ACDC pairs plus synthetic clean-target degradations"
    )
    p.add_argument("--data", default=r"E:\rgb_anon_trainvaltest\rgb_anon")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=500, help="Number of real training pairs to use (default: 500; 0 = use all)")
    p.add_argument("--synthetic-ratio", type=float, default=1.0,
                   help="synthetic examples per real example; 1.0 gives a 50/50 mix")
    p.add_argument("--out", default="checkpoints/acdc_v3_hybrid_best.pth")
    p.add_argument("--validation-images", default="outputs/hybrid_validation")
    args = p.parse_args()

    if args.size % 8 != 0:
        raise ValueError("--size must be divisible by 8")
    if args.synthetic_ratio < 0:
        raise ValueError("--synthetic-ratio cannot be negative")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    real_train = ACDCPairedDataset(args.data, "train", CONDITIONS, args.size, True, False)
    real_val = ACDCPairedDataset(args.data, "val", CONDITIONS, args.size, False, False)
    if args.limit and args.limit < len(real_train):
        real_train = Subset(real_train, list(range(args.limit)))

    synthetic_train_len = int(round(len(real_train) * args.synthetic_ratio))
    synthetic_val_len = max(len(real_val) // 1, 1) if args.synthetic_ratio > 0 else 0
    train_parts = [real_train]
    train_labels = ["real"]
    val_parts = [real_val]
    val_labels = ["real"]
    if synthetic_train_len:
        train_parts.append(SyntheticDegradationDataset(
            args.data, "train", args.size, length=synthetic_train_len
        ))
        train_labels.append("synthetic")
        val_parts.append(SyntheticDegradationDataset(
            args.data, "val", args.size, length=synthetic_val_len
        ))
        val_labels.append("synthetic")

    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_ds = ConcatDataset(val_parts) if len(val_parts) > 1 else val_parts[0]
    print(f"Real ACDC training pairs: {len(real_train)}")
    print(f"Synthetic training pairs: {synthetic_train_len}")
    print(f"Total training examples/epoch: {len(train_ds)}")
    print(f"Real ACDC validation pairs: {len(real_val)}")
    print(f"Synthetic validation examples: {synthetic_val_len}")
    print(f"Total validation examples: {len(val_ds)}")
    print(f"Patch size: {args.size}x{args.size}; batch size: {args.batch}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

    model = NightDehazeNet(base=24).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics, _ = run_epoch(
            model, train_loader, optimizer, scaler, device, train=True
        )
        train_loss = train_metrics["loss"]
        train_psnr = train_metrics["psnr"]
        train_ssim = train_metrics["ssim"]
        # Validate real ACDC pairs separately from synthetic pairs. This avoids
        # selecting a checkpoint just because one validation domain is easier.
        val_results = {}
        with torch.no_grad():
            for label, ds in zip(val_labels, val_parts):
                loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                                    num_workers=args.workers, pin_memory=device.type == "cuda")
                result = run_epoch(model, loader, optimizer, scaler, device, train=False)
                val_results[label] = result
        real_metrics, real_by_condition = val_results["real"]
        real_loss = real_metrics["loss"]
        real_psnr = real_metrics["psnr"]
        real_ssim = real_metrics["ssim"]
        if "synthetic" in val_results:
            syn_metrics, syn_by_condition = val_results["synthetic"]
        else:
            syn_metrics, syn_by_condition = None, {}
        # Use real validation loss to select the checkpoint; synthetic metrics
        # remain diagnostic and do not dominate model selection.
        selection_loss = real_loss
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | {elapsed:.1f}s | "
            f"train loss {train_loss:.4f} PSNR {train_psnr:.2f} SSIM {train_ssim:.4f} | "
            f"REAL val loss {real_loss:.4f} model PSNR {real_psnr:.2f} SSIM {real_ssim:.4f} | "
            f"input PSNR {real_metrics['input_psnr']:.2f} SSIM {real_metrics['input_ssim']:.4f}"
        )
        if syn_metrics is not None:
            print(f"  SYNTHETIC val loss {syn_metrics['loss']:.4f} | "
                  f"model PSNR {syn_metrics['psnr']:.2f} SSIM {syn_metrics['ssim']:.4f} | "
                  f"input PSNR {syn_metrics['input_psnr']:.2f} SSIM {syn_metrics['input_ssim']:.4f}")
        print("  Real validation by condition:")
        for name, metrics in sorted(real_by_condition.items()):
            print(f"    {name:18s} L1 {metrics['loss']:.4f} | "
                  f"model PSNR {metrics['psnr']:.2f} SSIM {metrics['ssim']:.4f} | "
                  f"input PSNR {metrics['input_psnr']:.2f} SSIM {metrics['input_ssim']:.4f}")
        if syn_by_condition:
            print("  Synthetic validation by condition:")
            for name, metrics in sorted(syn_by_condition.items()):
                print(f"    {name:18s} L1 {metrics['loss']:.4f} | "
                      f"model PSNR {metrics['psnr']:.2f} SSIM {metrics['ssim']:.4f} | "
                      f"input PSNR {metrics['input_psnr']:.2f} SSIM {metrics['input_ssim']:.4f}")

        if selection_loss < best:
            best = selection_loss
            torch.save({
                "model": model.state_dict(),
                "base": 24,
                "size": args.size,
                "dataset": "ACDC real pairs + synthetic clean-target haze/night/rain/snow/glare",
                "version": "v3-hybrid-synthetic",
                "epoch": epoch,
                "val_loss": real_loss,
                "val_psnr": real_psnr,
                "val_ssim": real_ssim,
                "real_validation_metrics": real_metrics,
                "real_validation_by_condition": real_by_condition,
                "synthetic_validation": {
                    "metrics": syn_metrics, "by_condition": syn_by_condition,
                },
            }, args.out)
            print(f"  saved best checkpoint (selected on REAL validation) -> {args.out}")
            # Save real and synthetic visual comparisons separately.
            with torch.no_grad():
                for label, ds in zip(val_labels, val_parts):
                    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                                        num_workers=args.workers, pin_memory=device.type == "cuda")
                    run_epoch(
                        model, loader, optimizer, scaler, device, train=False,
                        save_dir=os.path.join(args.validation_images, f"epoch_{epoch:03d}", label)
                    )

    print("Training complete.")
    print("Keep the existing app checkpoint until this model has been evaluated.")


if __name__ == "__main__":
    main()
