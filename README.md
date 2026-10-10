# NightVision-Dehaze v3 (hybrid training prototype)

**Nighttime and adverse-weather road-image restoration with CNN, channel attention and Transformer features**

This project combines real paired ACDC adverse-condition examples with synthetic degraded-clean examples. The synthetic examples are generated on the fly from clean reference images and use the original clean image as the training target. Synthetic effects are approximations, not exact models of real-world weather or glare.

## What the hybrid pipeline does

- Uses real ACDC fog, night, rain and snow input/reference pairs.
- Generates synthetic haze, low-light, rain, snow and glare inputs from clean reference images; each clean image is the target for its synthetic input.
- Keeps real ACDC inputs from being artificially degraded a second time by default.
- Uses L1 + SSIM + Sobel edge losses.
- Reports aggregate validation metrics and separate metrics for each real/synthetic validation category.
- Saves one validation montage per category (input | prediction | reference) when a new best checkpoint is found.
- Uses a separate v3 checkpoint filename so the existing v2 checkpoint is not overwritten.

## Dataset layout

Keep the ACDC dataset outside the repository. The expected root is:

```text
E:\rgb_anon_trainvaltest\rgb_anon
├── fog
│   ├── train / train_ref
│   ├── val / val_ref
│   └── test / test_ref
├── night
│   ├── train / train_ref
│   ├── val / val_ref
│   └── test / test_ref
├── rain
│   ├── train / train_ref
│   ├── val / val_ref
│   └── test / test_ref
└── snow
    ├── train / train_ref
    ├── val / val_ref
    └── test / test_ref
```

The paired loader matches `*_rgb_anon.png` input names with their corresponding `*_rgb_ref_anon.png` reference names. Synthetic training uses only `fog/train_ref` as its clean-image pool; synthetic validation uses only `fog/val_ref`. The test split is not used by the training script.

## Windows setup

Run these commands from the repository directory. If the virtual environment is in the parent project directory, activate it with the path below (adjust only if your folder differs):

```powershell
& "..\.venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Step 1 — small GPU smoke test

Start small on GPUs with limited VRAM. This trains on 40 real pairs plus 40 synthetic examples per epoch and evaluates on the validation split. It may still take time; watch GPU memory and temperature.

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 1 --limit 40 --batch 1 --size 128 --out "checkpoints\acdc_v3_smoke_best.pth" --validation-images "outputs\smoke_validation"
```

Check that it prints `Device: cuda`, finishes an epoch, saves a checkpoint and writes validation montages. If CUDA runs out of memory, retry with `--size 96` is not supported (size must be divisible by 8, so 96 is supported); or use `--size 96` and `--batch 1`.

## Step 2 — small experiment

Only after the smoke test succeeds:

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 3 --limit 200 --batch 1 --size 128 --out "checkpoints\acdc_v3_hybrid_best.pth"
```

## Step 3 — full hybrid training

The complete run uses all available real training pairs plus an equal number of synthetic examples per epoch. On a 2 GB GPU, use 128px patches and batch size 1 first; larger settings may exceed VRAM or take a long time.

```powershell
python train_acdc.py --data "E:\rgb_anon_trainvaltest\rgb_anon" --epochs 5 --limit 0 --batch 1 --size 128 --out "checkpoints\acdc_v3_hybrid_best.pth"
```

The script reports real and synthetic validation categories separately and saves the best checkpoint to the path supplied with `--out`. Validation montages are written under `outputs\hybrid_validation`. Each montage is ordered **input | model prediction | clean/reference target**.

## Run the Streamlit demo

```powershell
streamlit run app.py
```

The app currently loads its configured checkpoint path; do not assume it is using the newly trained v3 checkpoint until that path is deliberately updated and the model is evaluated. Keep the v2 checkpoint until v3 has been compared against it.

## Important evaluation notes

- Synthetic degradation is a controllable approximation. Strong results on synthetic data do not prove real-world robustness.
- Glare removal is inherently limited: saturated headlight regions may have lost information that a model cannot recover.
- Do not train on the ACDC test split. Report its metrics only after training choices are complete.
- This is a research/college prototype, not a guarantee of safety-critical road visibility.
- Keep the dataset outside GitHub and follow the dataset's license/terms.
