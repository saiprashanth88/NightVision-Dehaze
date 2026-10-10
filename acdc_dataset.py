import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".png", ".jpg", ".jpeg"}
SYNTHETIC_CONDITIONS = ("synthetic_haze", "synthetic_night", "synthetic_rain",
                        "synthetic_snow", "synthetic_glare")


class ACDCPairedDataset(Dataset):
    """Paired ACDC adverse-condition -> normal-condition dataset.

    Expected root: root/{fog,night,rain,snow}/{train,val,test} and matching
    *_ref folders. Real adverse inputs are paired with the supplied references.
    """

    def __init__(self, root, split="train", conditions=("fog", "night", "rain", "snow"),
                 size=256, augment=False, synthetic_glare=False):
        self.root = Path(root)
        self.split = split
        self.conditions = tuple(conditions)
        self.size = int(size)
        self.augment = augment
        self.synthetic_glare = synthetic_glare
        self.pairs = []

        for condition in self.conditions:
            input_dir = self.root / condition / split
            ref_dir = self.root / condition / f"{split}_ref"
            if not input_dir.exists():
                raise FileNotFoundError(f"Missing ACDC input directory: {input_dir}")
            if not ref_dir.exists():
                raise FileNotFoundError(f"Missing ACDC reference directory: {ref_dir}")

            refs = {}
            for p in ref_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    refs[p.name.replace("_rgb_ref_anon", "_rgb_anon")] = p

            for p in input_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS and p.name in refs:
                    self.pairs.append((p, refs[p.name], condition))

        if not self.pairs:
            raise RuntimeError(
                f"No ACDC pairs found under {self.root} for split={split}, conditions={conditions}"
            )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _read(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _crop_pair(self, x, y):
        h, w = x.shape[:2]
        s = self.size
        if h < s or w < s:
            scale = max(s / h, s / w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
            y = cv2.resize(y, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = x.shape[:2]

        if self.split == "train":
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
        else:
            top = max(0, (h - s) // 2)
            left = max(0, (w - s) // 2)
        return x[top:top + s, left:left + s], y[top:top + s, left:left + s]

    def _degrade(self, x):
        """Mild sensor/illumination augmentation for the real adverse input."""
        if random.random() < 0.55:
            gamma = random.uniform(0.80, 1.12)
            x = np.power(np.clip(x, 0, 1), gamma)
        if random.random() < 0.25:
            x *= random.uniform(0.82, 0.98)
        if random.random() < 0.20:
            noise = np.random.normal(0, random.uniform(0.002, 0.008), x.shape).astype(np.float32)
            x = np.clip(x + noise, 0, 1)
        if self.synthetic_glare and random.random() < 0.20:
            h, w = x.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = random.randrange(w), random.randrange(h)
            sigma = random.uniform(max(4, min(h, w) * 0.025), max(6, min(h, w) * 0.09))
            glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
            x = np.clip(x + glow[..., None] * random.uniform(0.05, 0.18), 0, 1)
        return x

    def __getitem__(self, index):
        inp_path, ref_path, condition = self.pairs[index]
        x = self._read(inp_path).astype(np.float32) / 255.0
        y = self._read(ref_path).astype(np.float32) / 255.0
        x, y = self._crop_pair(x, y)

        if self.augment:
            if random.random() < 0.5:
                x = np.ascontiguousarray(x[:, ::-1])
                y = np.ascontiguousarray(y[:, ::-1])
            # Only add extra degradations when explicitly requested. Real ACDC
            # inputs should not routinely be degraded a second time.
            if self.synthetic_glare:
                x = self._degrade(x)

        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)))
        return x, y, condition, str(inp_path)


class SyntheticDegradationDataset(Dataset):
    """On-the-fly degraded-clean -> original-clean training examples.

    Uses only one split's fog reference directory as the clean-image source to
    avoid multiplying identical reference scenes across all four conditions.
    Training must use split='train'; validation uses split='val'.
    """

    def __init__(self, root, split="train", size=256, length=None):
        self.root = Path(root)
        self.split = split
        self.size = int(size)
        source = self.root / "fog" / f"{split}_ref"
        if not source.exists():
            raise FileNotFoundError(f"Missing clean-image source directory: {source}")
        self.images = sorted(p for p in source.rglob("*")
                             if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not self.images:
            raise RuntimeError(f"No clean reference images found in {source}")
        self.length = int(length) if length is not None else len(self.images)

    def __len__(self):
        return self.length

    def _degrade(self, clean, kind, rng):
        x = clean.copy()
        h, w = x.shape[:2]
        if kind == "synthetic_haze":
            # Smooth spatial transmission plus veiling light and reduced contrast.
            coarse_h, coarse_w = max(2, h // 24), max(2, w // 24)
            small = rng.uniform(0.22, 0.62, size=(coarse_h, coarse_w)).astype(np.float32)
            transmission = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            transmission = cv2.GaussianBlur(transmission, (0, 0), max(2, min(h, w) * 0.06))
            transmission = np.clip(transmission, 0.22, 0.68)[..., None]
            airlight = np.array([0.76, 0.80, 0.84], dtype=np.float32)
            x = x * transmission + airlight * (1.0 - transmission)
            x = np.clip(x + rng.normal(0, 0.004, x.shape).astype(np.float32), 0, 1)
        elif kind == "synthetic_night":
            gamma = float(rng.uniform(1.8, 2.8))
            x = np.power(np.clip(x, 0, 1), gamma)
            x *= rng.uniform(0.28, 0.62)
            # Low-light color cast, sensor noise and a small amount of chroma variation.
            x[..., 0] *= rng.uniform(0.82, 1.0)
            x[..., 2] *= rng.uniform(0.78, 0.98)
            x = np.clip(x + rng.normal(0, 0.012, x.shape).astype(np.float32), 0, 1)
        elif kind == "synthetic_rain":
            # Streaks at a slant, a slight contrast veil, and sensor noise.
            x *= float(rng.uniform(0.62, 0.86))
            overlay = np.zeros((h, w), dtype=np.float32)
            count = max(35, (h * w) // 100)
            for _ in range(count):
                px, py = int(rng.integers(0, w)), int(rng.integers(0, h))
                length = int(rng.integers(5, max(6, min(22, h // 2))))
                dx = int(rng.integers(-3, 4))
                cv2.line(overlay, (px, py),
                         (min(w - 1, max(0, px + dx)), min(h - 1, py + length)),
                         float(rng.uniform(0.30, 0.85)), 1, cv2.LINE_AA)
            x = np.clip(x * 0.92 + 0.08 + overlay[..., None] * 0.55, 0, 1)
        elif kind == "synthetic_snow":
            # Combine smaller flakes, a few larger out-of-focus flakes, and a light veil.
            overlay = np.zeros((h, w), dtype=np.float32)
            count = max(18, (h * w) // 220)
            for _ in range(count):
                px, py = int(rng.integers(0, w)), int(rng.integers(0, h))
                radius = int(rng.choice([1, 1, 1, 2, 3]))
                cv2.circle(overlay, (px, py), radius, float(rng.uniform(0.65, 1.0)), -1, cv2.LINE_AA)
            overlay = cv2.GaussianBlur(overlay, (3, 3), 0.45)
            x = np.clip(x * 0.80 + overlay[..., None] * 0.62, 0, 1)
            x = np.clip(x * 0.90 + np.array([0.10, 0.11, 0.13], dtype=np.float32), 0, 1)
        elif kind == "synthetic_glare":
            # Broad veiling bloom + saturated cores; deliberately stronger than v3,
            # but still a synthetic approximation and not a substitute for real glare data.
            yy, xx = np.mgrid[0:h, 0:w]
            x *= float(rng.uniform(0.78, 0.94))
            for _ in range(int(rng.integers(1, 3))):
                cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
                sigma = float(rng.uniform(max(7, min(h, w) * 0.045), max(10, min(h, w) * 0.13)))
                distance2 = (xx - cx) ** 2 + (yy - cy) ** 2
                glow = np.exp(-distance2 / (2 * sigma * sigma))
                core = np.exp(-distance2 / (2 * (sigma * 0.18) ** 2))
                veil = np.clip(glow * rng.uniform(0.28, 0.55), 0, 1)
                x = x * (1.0 - veil[..., None] * 0.45) + veil[..., None] * np.array(
                    [0.95, 0.92, 0.86], dtype=np.float32
                )
                x = np.clip(x + core[..., None] * 0.9, 0, 1)
        return np.ascontiguousarray(np.clip(x, 0, 1), dtype=np.float32)

    def __getitem__(self, index):
        path = self.images[index % len(self.images)]
        clean = ACDCPairedDataset._read(path).astype(np.float32) / 255.0
        # Use a deterministic validation crop; training gets a random crop.
        h, w = clean.shape[:2]
        s = self.size
        if h < s or w < s:
            scale = max(s / h, s / w)
            clean = cv2.resize(clean, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            h, w = clean.shape[:2]
        if self.split == "train":
            top, left = random.randint(0, h - s), random.randint(0, w - s)
        else:
            top, left = (h - s) // 2, (w - s) // 2
        clean = clean[top:top + s, left:left + s]
        if self.split == "train" and random.random() < 0.5:
            clean = np.ascontiguousarray(clean[:, ::-1])

        kind = SYNTHETIC_CONDITIONS[index % len(SYNTHETIC_CONDITIONS)]
        # Fixed seed per item makes synthetic validation repeatable.
        rng = np.random.default_rng(index + (0 if self.split == "train" else 100000))
        degraded = self._degrade(clean, kind, rng)
        x = torch.from_numpy(np.ascontiguousarray(degraded.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1)))
        return x, y, kind, str(path)
