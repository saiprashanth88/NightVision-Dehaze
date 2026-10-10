import os
import io

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import NightDehazeNet
from inference import enhance_tiled, safe_quality_guard, laplacian_sharpness

st.set_page_config(
    page_title="NightVision-Dehaze | Nighttime Visibility Analysis",
    page_icon="🌙",
    layout="wide",
)

CHECKPOINT = "checkpoints/acdc_v2_best.pth"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CHECKPOINT):
        return None, device, 256, "Checkpoint not found"
    try:
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model = NightDehazeNet(base=ckpt.get("base", 24)).to(device).float()
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, device, int(ckpt.get("size", 256)), "Loaded"
    except Exception as exc:
        return None, device, 256, f"Could not load checkpoint: {exc}"


def to_float(rgb):
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def to_uint8(rgb01):
    return np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)


def preprocess_image(rgb):
    """Light, conservative denoising and normalization for visualization."""
    denoised = cv2.fastNlMeansDenoisingColored(rgb, None, 3, 3, 7, 21)
    # Keep this stage conservative: don't aggressively brighten the source.
    return denoised


def condition_analysis(rgb):
    """Heuristic visual indicators, not a trained condition classifier."""
    f = to_float(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    mean_luma = float(np.mean(gray_f))
    contrast = float(np.std(gray_f))
    saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    bright_fraction = float(np.mean(gray >= 245))
    dark_fraction = float(np.mean(gray <= 45))

    # A rough fog/haze cue: low contrast with limited color saturation.
    fog_score = float(np.clip((0.22 - contrast) / 0.18 * 0.65 + (0.38 - saturation) * 0.35, 0, 1))
    low_light_score = float(np.clip((0.42 - mean_luma) / 0.38, 0, 1))
    glare_score = float(np.clip(bright_fraction * 18.0 + max(0.0, mean_luma - 0.55) * 0.35, 0, 1))

    # Simple explanatory map: blue = darker pixels, green = mid-tones,
    # red = very bright pixels. It is a visualization, not semantic segmentation.
    dark = np.clip((0.45 - gray_f) / 0.45, 0, 1)
    bright = np.clip((gray_f - 0.78) / 0.22, 0, 1)
    mid = np.clip(1.0 - dark - bright, 0, 1)
    heat = np.stack([bright, mid * 0.75, dark], axis=-1)
    heat = np.clip(heat * 255, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb, 0.48, heat, 0.52, 0)

    def level(score):
        if score >= 0.66:
            return "High"
        if score >= 0.34:
            return "Moderate"
        return "Low"

    return {
        "fog_score": fog_score,
        "low_light_score": low_light_score,
        "glare_score": glare_score,
        "brightness": mean_luma,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": laplacian_sharpness(f),
        "bright_fraction": bright_fraction,
        "dark_fraction": dark_fraction,
        "fog_label": level(fog_score),
        "low_light_label": "Detected" if low_light_score >= 0.34 else "Low",
        "glare_label": "Detected" if glare_score >= 0.34 else "Low",
        "overlay": overlay,
    }


def classic_enhancement(rgb, strength=0.45):
    """Fallback when the trained checkpoint is unavailable; not a learned model."""
    f = to_float(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

    # Gentle gamma lift for dark scenes, limited to avoid washing out highlights.
    gray_mean = float(np.mean(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))) / 255.0
    gamma = 0.82 if gray_mean < 0.38 else 0.94
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    lifted = cv2.LUT(contrast, lut)

    blur = cv2.GaussianBlur(lifted, (0, 0), 1.0)
    sharp = cv2.addWeighted(lifted, 1.12, blur, -0.12, 0)
    alpha = float(np.clip(strength, 0, 1))
    out = cv2.addWeighted(rgb, 1.0 - alpha, sharp, alpha, 0)
    return to_float(out)


def postprocess_image(rgb01, strength=0.45):
    img = to_uint8(rgb01)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrast, 1.10, blur, -0.10, 0)
    alpha = float(np.clip(strength, 0, 1))
    return to_float(cv2.addWeighted(img, 1.0 - alpha, sharp, alpha, 0))


def image_metrics(rgb01):
    gray = cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return {
        "Sharpness (Laplacian variance)": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "Contrast (gray std.)": float(np.std(gray)),
        "Mean brightness": float(np.mean(gray)),
    }


def reference_metrics(candidate, reference):
    if candidate.shape[:2] != reference.shape[:2]:
        reference = cv2.resize(reference, (candidate.shape[1], candidate.shape[0]), interpolation=cv2.INTER_AREA)
    cand = to_float(candidate)
    ref = to_float(reference)
    return (
        float(peak_signal_noise_ratio(ref, cand, data_range=1.0)),
        float(structural_similarity(ref, cand, channel_axis=2, data_range=1.0)),
    )


def png_bytes(rgb01):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(to_uint8(rgb01), cv2.COLOR_RGB2BGR))
    return encoded.tobytes() if ok else b""


def show_metric_graphs(input_rgb, output_rgb):
    """Colorful, easy-to-read input-vs-output charts; no sidebar reference upload needed."""
    st.markdown("## 📊 Input vs. Enhanced Output — Visual Comparison")

    input01 = to_float(input_rgb)
    output01 = to_float(output_rgb)
    before = image_metrics(input01)
    after = image_metrics(output01)

    # Extra interpretable image statistics.
    before_gray = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    after_gray = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    before_brightness = float(before_gray.mean())
    after_brightness = float(after_gray.mean())
    before_contrast = float(before_gray.std())
    after_contrast = float(after_gray.std())
    before_saturation = float(cv2.cvtColor(input_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].mean())
    after_saturation = float(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2HSV)[:, :, 1].mean())

    labels = ["Input", "Enhanced output"]
    palette = ["#3498db", "#2ecc71"]

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        bars = ax.bar(labels, [before["Sharpness (Laplacian variance)"], after["Sharpness (Laplacian variance)"]], color=palette, width=0.58)
        ax.set_title("Sharpness comparison", fontweight="bold")
        ax.set_ylabel("Laplacian variance")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        bars = ax.bar(labels, [before_contrast, after_contrast], color=palette, width=0.58)
        ax.set_title("Contrast comparison", fontweight="bold")
        ax.set_ylabel("Grayscale standard deviation (0–1)")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        bars = ax.bar(labels, [before_brightness, after_brightness], color=["#f39c12", "#8e44ad"], width=0.58)
        ax.set_title("Brightness comparison", fontweight="bold")
        ax.set_ylabel("Mean grayscale brightness (0–1)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        bars = ax.bar(labels, [before_saturation, after_saturation], color=["#e67e22", "#16a085"], width=0.58)
        ax.set_title("Color saturation comparison", fontweight="bold")
        ax.set_ylabel("Mean HSV saturation (0–255)")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # These scores describe similarity between output and input, not correctness
    # against a clean ground-truth image.
    similarity_psnr, similarity_ssim = reference_metrics(output_rgb, input_rgb)
    st.markdown("### Input–output similarity")
    c1, c2, c3 = st.columns(3)
    c1.metric("PSNR (output vs input)", f"{similarity_psnr:.2f} dB")
    c2.metric("SSIM (output vs input)", f"{similarity_ssim:.3f}")
    c3.metric("Brightness change", f"{after_brightness - before_brightness:+.3f}")
    st.caption(
        "PSNR and SSIM here compare the enhanced result with the original input only. "
        "They measure how much the image changed, not whether the restored image is objectively correct. "
        "Ground-truth restoration-quality evaluation requires a matching clean reference image."
    )

st.title("🌙 Nighttime Visibility Analysis")
st.caption("Night Time Defogging Vision Network with Smart Glare Suppression")
st.markdown(
    "Upload one or more road images to see the processing pipeline, heuristic fog/low-light/glare indicators, "
    "final enhancement, and image-quality graphs."
)

model, device, tile_size, model_status = load_model()
with st.sidebar:
    st.header("Analysis settings")
    st.write(f"**Device:** {'GPU' if device.type == 'cuda' else 'CPU'}")
    st.write(f"**V2 model:** {model_status}")
    if model is None:
        st.warning("The trained checkpoint is not available. The app will use a classical OpenCV fallback for the demo.")
        st.caption("The fallback is not your trained CNN + Transformer model.")
    strength = st.slider("Post-processing strength", 0.0, 1.0, 0.45, 0.05)
    show_intermediate = st.checkbox("Show all processing stages", value=True)

uploads = st.file_uploader(
    "Upload nighttime / foggy / rainy / snowy road images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="road_images",
)

if not uploads:
    st.info("Upload one or more images to generate the full analysis screen. You can test the dashboard without training; the OpenCV fallback will be used if the V2 checkpoint is missing.")
    st.markdown("#### Dashboard preview")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Fog detection", "Awaiting image")
    p2.metric("Low-light", "Awaiting image")
    p3.metric("Glare", "Awaiting image")
    p4.metric("Image clarity", "Awaiting image")
else:
    for index, uploaded in enumerate(uploads):
        try:
            original_rgb = np.array(Image.open(uploaded).convert("RGB"))
        except Exception as exc:
            st.error(f"Could not read {uploaded.name}: {exc}")
            continue

        original01 = to_float(original_rgb)
        preprocessed = preprocess_image(original_rgb)
        condition = condition_analysis(original_rgb)

        with st.expander(f"Image {index + 1}: {uploaded.name}", expanded=(index == 0)):
            st.subheader("Nighttime Visibility Analysis")
            st.caption(f"Image size: {original_rgb.shape[1]} × {original_rgb.shape[0]} px")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fog / haze cue", condition["fog_label"], f'{condition["fog_score"] * 100:.0f}% heuristic score')
            m2.metric("Low-light cue", condition["low_light_label"], f'{condition["low_light_score"] * 100:.0f}% heuristic score')
            m3.metric("Bright glare cue", condition["glare_label"], f'{condition["glare_score"] * 100:.0f}% heuristic score')
            m4.metric("Image clarity", f'{condition["sharpness"]:.1f}', "Laplacian sharpness")

            with st.spinner(f"Processing {uploaded.name}..."):
                pre01 = to_float(preprocessed)
                if model is not None:
                    try:
                        raw_model = enhance_tiled(model, preprocessed, device, tile=tile_size, overlap=min(64, tile_size // 4))
                        model_out, original_sharp, model_sharp, mean_change, guarded = safe_quality_guard(pre01, raw_model)
                        method_label = "Trained V2 CNN + Channel Attention + Transformer"
                    except Exception as exc:
                        st.warning(f"V2 inference failed for this image; using OpenCV fallback. Details: {exc}")
                        model_out = classic_enhancement(preprocessed, strength)
                        original_sharp = laplacian_sharpness(pre01)
                        model_sharp = laplacian_sharpness(model_out)
                        mean_change = float(np.mean(np.abs(model_out - pre01)))
                        guarded = False
                        method_label = "OpenCV fallback (V2 inference failed)"
                else:
                    model_out = classic_enhancement(preprocessed, strength)
                    original_sharp = laplacian_sharpness(pre01)
                    model_sharp = laplacian_sharpness(model_out)
                    mean_change = float(np.mean(np.abs(model_out - pre01)))
                    guarded = False
                    method_label = "OpenCV classical enhancement fallback"

                final = postprocess_image(model_out, strength)
                stages = {
                    "Input": original01,
                    "Preprocessed": pre01,
                    "CNN + Attention + Transformer" if model is not None and "fallback" not in method_label.lower() else "Enhancement stage": model_out,
                    "Final output": final,
                }

            st.markdown(f"**Processing method:** {method_label}")
            if guarded:
                st.warning("The quality guard blended the model output toward the preprocessed input because a large change or sharpness drop was detected.")
            elif model is None:
                st.warning("The trained checkpoint was not loaded; this result is produced by classical image processing, not by the trained neural network.")

            if show_intermediate:
                columns = st.columns(4)
                columns[0].image(original_rgb, caption="1. Input image", width="stretch")
                columns[1].image(preprocessed, caption="2. Preprocessing (denoise)", width="stretch")
                columns[2].image(condition["overlay"], caption="3. Condition-analysis visualization", width="stretch")
                columns[3].image(to_uint8(model_out), caption="4. Neural / fallback enhancement stage", width="stretch")
                st.markdown("#### Final enhanced output")
                left, right = st.columns([1.5, 1])
                with left:
                    st.image(to_uint8(final), caption="Final output (contrast and sharpening refinement)", width="stretch")
                with right:
                    st.markdown("**Condition indicators**")
                    st.progress(condition["fog_score"], text=f'Fog / haze cue: {condition["fog_score"] * 100:.0f}%')
                    st.progress(condition["low_light_score"], text=f'Low-light cue: {condition["low_light_score"] * 100:.0f}%')
                    st.progress(condition["glare_score"], text=f'Bright-glare cue: {condition["glare_score"] * 100:.0f}%')
                    st.caption("These indicators are simple image-statistic heuristics, not a validated detection model.")
            else:
                left, right = st.columns(2)
                left.image(original_rgb, caption="Original input", width="stretch")
                right.image(to_uint8(final), caption="Final enhanced output", width="stretch")

            sharp_a, sharp_b, delta = st.columns(3)
            sharp_a.metric("Input sharpness", f"{original_sharp:.2f}")
            sharp_b.metric("Enhancement-stage sharpness", f"{model_sharp:.2f}")
            delta.metric("Mean pixel change", f"{mean_change:.4f}")

            show_metric_graphs(original_rgb, to_uint8(final))

            st.download_button(
                "Download final enhanced image",
                data=png_bytes(final),
                file_name=f"{os.path.splitext(uploaded.name)[0]}_nightvision_enhanced.png",
                mime="image/png",
                key=f"download_{index}_{uploaded.name}",
            )

st.markdown("---")
st.caption(
    "Research/demo prototype. Brightness, fog and glare indicators are heuristic estimates. "
    "Estimated visibility in metres is intentionally not displayed because this app has no calibrated distance-estimation model."
)
