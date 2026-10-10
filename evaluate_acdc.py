import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from acdc_dataset import ACDCPairedDataset
from model import NightDehazeNet


def psnr_per_image(pred, target):
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_per_image(pred, target, window=7):
    pad = window // 2
    mx = F.avg_pool2d(pred, window, 1, pad)
    my = F.avg_pool2d(target, window, 1, pad)
    vx = F.avg_pool2d(pred * pred, window, 1, pad) - mx * mx
    vy = F.avg_pool2d(target * target, window, 1, pad) - my * my
    cov = F.avg_pool2d(pred * target, window, 1, pad) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx * mx + my * my + c1) * (vx + vy + c2) + 1e-8
    )
    return score.clamp(0, 1).mean(dim=(1, 2, 3))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NightVision-Dehaze checkpoint against the real ACDC validation inputs."
    )
    parser.add_argument("--data", default=r"E:\rgb_anon_trainvaltest\rgb_anon")
    parser.add_argument(
        "--checkpoint", default="checkpoints/acdc_v3_experiment_best.pth"
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum number of validation pairs to evaluate; 0 evaluates all pairs."
    )
    args = parser.parse_args()

    if args.size <= 0 or args.size % 8 != 0:
        raise ValueError("--size must be positive and divisible by 8.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.resolve()}\n"
            "Check --checkpoint or copy the trained checkpoint into this repository."
        )

    # Checkpoints produced by train_acdc.py contain a 'model' state dict.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "Unexpected checkpoint format: expected a dictionary containing a 'model' state dict."
        )

    model = NightDehazeNet(base=int(checkpoint.get("base", 24))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Checkpoint: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    dataset = ACDCPairedDataset(
        args.data,
        split="val",
        conditions=("fog", "night", "rain", "snow"),
        size=args.size,
        augment=False,
        synthetic_glare=False,
    )
    if args.limit > 0:
        dataset.pairs = dataset.pairs[:args.limit]
    if len(dataset) == 0:
        raise RuntimeError("No validation pairs found. Check the --data directory.")

    print(f"Validation pairs: {len(dataset)}")
    print(f"Evaluation crop: {args.size}x{args.size}; batch size: 1")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    stats = defaultdict(lambda: {
        "count": 0,
        "input_psnr": 0.0,
        "model_psnr": 0.0,
        "input_ssim": 0.0,
        "model_ssim": 0.0,
    })

    with torch.inference_mode():
        for index, (x, y, condition, _paths) in enumerate(loader, start=1):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            output = model(x).clamp(0, 1)
            name = condition[0]

            values = {
                "input_psnr": psnr_per_image(x, y),
                "model_psnr": psnr_per_image(output, y),
                "input_ssim": ssim_per_image(x, y),
                "model_ssim": ssim_per_image(output, y),
            }
            row = stats[name]
            row["count"] += 1
            for key, value in values.items():
                row[key] += float(value[0].item())

            if index % 100 == 0 or index == len(dataset):
                print(f"Evaluated {index}/{len(dataset)} pairs")

    metric_keys = ("input_psnr", "model_psnr", "input_ssim", "model_ssim")
    totals = {key: 0.0 for key in metric_keys}
    total_count = 0

    print("\nREAL ACDC VALIDATION RESULTS")
    print("-" * 86)
    print(
        f"{'Condition':<12} {'Images':>7} {'Input PSNR':>12} "
        f"{'Model PSNR':>12} {'Input SSIM':>12} {'Model SSIM':>12}"
    )
    print("-" * 86)

    for name in ("fog", "night", "rain", "snow"):
        if name not in stats:
            continue
        row = stats[name]
        n = row["count"]
        print(
            f"{name:<12} {n:>7} "
            f"{row['input_psnr'] / n:>12.3f} "
            f"{row['model_psnr'] / n:>12.3f} "
            f"{row['input_ssim'] / n:>12.4f} "
            f"{row['model_ssim'] / n:>12.4f}"
        )
        total_count += n
        for key in metric_keys:
            totals[key] += row[key]

    if total_count:
        print("-" * 86)
        averages = {key: totals[key] / total_count for key in metric_keys}
        print(
            f"{'OVERALL':<12} {total_count:>7} "
            f"{averages['input_psnr']:>12.3f} "
            f"{averages['model_psnr']:>12.3f} "
            f"{averages['input_ssim']:>12.4f} "
            f"{averages['model_ssim']:>12.4f}"
        )
        print("-" * 86)
        print(f"PSNR change (model - input): {averages['model_psnr'] - averages['input_psnr']:+.3f} dB")
        print(f"SSIM change (model - input): {averages['model_ssim'] - averages['input_ssim']:+.4f}")
        print("Positive differences indicate improvement on average; negative differences indicate degradation.")
        print("Note: results use the same center-cropped patches as validation, not full-resolution images.")


if __name__ == "__main__":
    main()
