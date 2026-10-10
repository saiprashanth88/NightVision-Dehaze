import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from acdc_dataset import ACDCPairedDataset
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
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray_x, kx, padding=1)
    gy = F.conv2d(gray_x, ky, padding=1)
    gx_t = F.conv2d(gray_y, kx, padding=1)
    gy_t = F.conv2d(gray_y, ky, padding=1)
    return F.l1_loss(gx, gx_t) + F.l1_loss(gy, gy_t)


def psnr(x, y):
    mse = F.mse_loss(x, y).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * torch.log10(torch.tensor(1.0 / mse, device=x.device)).item()


def run_epoch(model, loader, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss = total_psnr = total_ssim = n = 0
    for x, y, _, _ in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(x)
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
        total_loss += loss.item() * bs
        total_psnr += psnr(out.detach(), y) * bs
        total_ssim += (1.0 - sl.item()) * bs
        n += bs
    return total_loss / n, total_psnr / n, total_ssim / n


def main():
    p = argparse.ArgumentParser(description="Train NightVision-Dehaze v2 on paired ACDC conditions")
    p.add_argument("--data", default=r"E:\rgb_anon_trainvaltest\rgb_anon")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = use all training pairs")
    p.add_argument("--out", default="checkpoints/acdc_v2_best.pth")
    args = p.parse_args()

    if args.size % 8 != 0:
        raise ValueError("--size must be divisible by 8")

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds = ACDCPairedDataset(args.data, "train", CONDITIONS, args.size, True, True)
    val_ds = ACDCPairedDataset(args.data, "val", CONDITIONS, args.size, False, False)
    if args.limit and args.limit < len(train_ds):
        train_ds = Subset(train_ds, list(range(args.limit)))

    print(f"Conditions: {', '.join(CONDITIONS)}")
    print(f"Training pairs: {len(train_ds)}")
    print(f"Validation pairs: {len(val_ds)}")
    print(f"Patch size: {args.size}x{args.size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")

    model = NightDehazeNet(base=24).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_psnr, train_ssim = run_epoch(model, train_loader, optimizer, scaler, device, True)
        with torch.no_grad():
            val_loss, val_psnr, val_ssim = run_epoch(model, val_loader, optimizer, scaler, device, False)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | {elapsed:.1f}s | "
            f"train loss {train_loss:.4f} PSNR {train_psnr:.2f} SSIM {train_ssim:.4f} | "
            f"val loss {val_loss:.4f} PSNR {val_psnr:.2f} SSIM {val_ssim:.4f}"
        )
        if val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(),
                "base": 24,
                "size": args.size,
                "dataset": "ACDC fog + night + rain + snow paired restoration",
                "version": "v2-patch-edge-loss",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, args.out)
            print(f"  saved best checkpoint -> {args.out}")

    print("Training complete.")


if __name__ == "__main__":
    main()
