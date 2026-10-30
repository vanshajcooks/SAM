
"""
03_unet_evaluate.py

Evaluate the trained U-Net on:
    Dev: threshold selection
    E: final held-out evaluation

No training or checkpoint modification occurs here.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\SAM")

DATA_DIR = PROJECT_DIR / "data"
MANIFEST_DIR = PROJECT_DIR / "manifests"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "unet"

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
SPLIT_MANIFEST = MANIFEST_DIR / "split_manifest_v2.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / "unet_best.pth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEIGHT = 256
WIDTH = 1600
CROP_WIDTH = 512
CROP_STARTS = (0, 363, 726, 1088)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. LOAD DATA AND CHECKPOINT
# ============================================================

print("=" * 60)
print("U-NET EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

df = pd.read_csv(TRAIN_CSV)

with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)

DEV = list(manifest["splits"]["dev"])
TEST_E = list(manifest["splits"]["test_E"])

rles_by_image = (
    df.groupby("ImageId")["EncodedPixels"]
    .apply(list)
    .to_dict()
)

assert len(DEV) == 200
assert len(TEST_E) == 1200

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(CHECKPOINT_PATH)


# ============================================================
# 3. RLE DECODING
# ============================================================

def decode_rle(encoded_pixels):

    values = np.asarray(
        [int(x) for x in encoded_pixels.split()],
        dtype=np.int64
    )

    if len(values) == 0 or len(values) % 2 != 0:
        raise ValueError("Invalid RLE sequence")

    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths

    if (
        np.any(starts < 0)
        or np.any(lengths <= 0)
        or np.any(ends > HEIGHT * WIDTH)
    ):
        raise ValueError("RLE outside image boundaries")

    flat = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)

    for start, end in zip(starts, ends):
        flat[start:end] = 1

    return flat.reshape((HEIGHT, WIDTH), order="F")


def load_image_and_mask(image_id):

    with Image.open(TRAIN_IMAGES_DIR / image_id) as img:
        image = np.asarray(img.convert("RGB"), dtype=np.uint8)

    if image.shape != (HEIGHT, WIDTH, 3):
        raise ValueError(
            f"Unexpected image shape: {image.shape}"
        )

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for rle in rles_by_image.get(image_id, []):
        mask |= decode_rle(rle)

    return image, mask


# ============================================================
# 4. U-NET ARCHITECTURE
# Must match Script 02 exactly.
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


print("\n========== LOADING CHECKPOINT ==========")

model = UNet().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded:", CHECKPOINT_PATH)
print("Best training epoch:", checkpoint["epoch"])
print(
    "Saved development Dice:",
    round(checkpoint["best_dev_present_dice"], 4)
)


# ============================================================
# 5. FULL-WIDTH PROBABILITY PREDICTION
# ============================================================

def predict_full_width(image):

    probability_sum = np.zeros(
        (HEIGHT, WIDTH), dtype=np.float32
    )

    count_sum = np.zeros(
        (HEIGHT, WIDTH), dtype=np.float32
    )

    with torch.no_grad():

        for left in CROP_STARTS:

            crop = image[:, left:left + CROP_WIDTH, :]

            x = (
                torch.from_numpy(
                    crop.astype(np.float32).copy() / 255.0
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            probability_sum[
                :, left:left + CROP_WIDTH
            ] += probs

            count_sum[
                :, left:left + CROP_WIDTH
            ] += 1

    return probability_sum / np.maximum(count_sum, 1)


# ============================================================
# 6. METRIC ACCUMULATION
# ============================================================

def new_metric_bucket():

    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "present_dice_sum": 0.0,
        "present_iou_sum": 0.0,
        "present_count": 0,
        "empty_fp_pixels": [],
        "empty_false_positive_count": 0,
        "empty_count": 0,
        "all_dice_sum": 0.0,
        "all_iou_sum": 0.0,
        "image_count": 0,
    }


def update_metrics(bucket, pred_mask, gt_mask):

    intersection = int(
        np.logical_and(pred_mask, gt_mask).sum()
    )

    pred_pixels = int(pred_mask.sum())
    gt_pixels = int(gt_mask.sum())

    union = int(
        np.logical_or(pred_mask, gt_mask).sum()
    )

    dice = (
        2 * intersection / (pred_pixels + gt_pixels)
        if pred_pixels + gt_pixels > 0 else 1.0
    )

    iou = (
        intersection / union
        if union > 0 else 1.0
    )

    bucket["all_dice_sum"] += dice
    bucket["all_iou_sum"] += iou
    bucket["image_count"] += 1

    bucket["tp"] += intersection
    bucket["fp"] += pred_pixels - intersection
    bucket["fn"] += gt_pixels - intersection

    if gt_pixels > 0:

        bucket["present_dice_sum"] += dice
        bucket["present_iou_sum"] += iou
        bucket["present_count"] += 1

    else:

        bucket["empty_fp_pixels"].append(pred_pixels)
        bucket["empty_count"] += 1

        if pred_pixels > 0:
            bucket["empty_false_positive_count"] += 1


def finalize_metrics(bucket):

    tp = bucket["tp"]
    fp = bucket["fp"]
    fn = bucket["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    global_dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 1.0
    )

    global_iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 1.0
    )

    return {
        "images": bucket["image_count"],
        "defect_present_images": bucket["present_count"],
        "defect_free_images": bucket["empty_count"],

        "mean_present_dice": (
            bucket["present_dice_sum"]
            / max(bucket["present_count"], 1)
        ),

        "mean_present_iou": (
            bucket["present_iou_sum"]
            / max(bucket["present_count"], 1)
        ),

        "global_pixel_dice": global_dice,
        "global_pixel_iou": global_iou,

        "precision": precision,
        "recall": recall,

        "mean_empty_fp_pixels": (
            float(np.mean(bucket["empty_fp_pixels"]))
            if bucket["empty_fp_pixels"] else 0.0
        ),

        "empty_image_false_positive_rate": (
            bucket["empty_false_positive_count"]
            / max(bucket["empty_count"], 1)
        ),

        "mean_all_image_dice": (
            bucket["all_dice_sum"]
            / max(bucket["image_count"], 1)
        ),

        "mean_all_image_iou": (
            bucket["all_iou_sum"]
            / max(bucket["image_count"], 1)
        ),
    }


# ============================================================
# 7. THRESHOLD SELECTION ON DEV ONLY
# ============================================================

print("\n========== DEV THRESHOLD SELECTION ==========")

dev_buckets = {
    threshold: new_metric_bucket()
    for threshold in THRESHOLDS
}

with torch.no_grad():

    for index, image_id in enumerate(DEV, start=1):

        image, gt_mask = load_image_and_mask(image_id)
        probabilities = predict_full_width(image)

        for threshold in THRESHOLDS:

            pred_mask = probabilities >= threshold

            update_metrics(
                dev_buckets[threshold],
                pred_mask,
                gt_mask
            )

        if index % 25 == 0:
            print(f"Processed Dev images: {index}/{len(DEV)}")


dev_results = {}

print("\nDEV THRESHOLD RESULTS")
print("-" * 90)
print(
    "Threshold | Present Dice | Present IoU | "
    "Precision | Recall | Empty FP Pixels"
)

for threshold in THRESHOLDS:

    metrics = finalize_metrics(dev_buckets[threshold])
    dev_results[str(threshold)] = metrics

    print(
        f"{threshold:>9.1f} | "
        f"{metrics['mean_present_dice']:.4f} | "
        f"{metrics['mean_present_iou']:.4f} | "
        f"{metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | "
        f"{metrics['mean_empty_fp_pixels']:.0f}"
    )


# Select maximum defect-present Dice.
# Ties are resolved in favor of the higher threshold.
best_threshold = max(
    THRESHOLDS,
    key=lambda t: (
        dev_results[str(t)]["mean_present_dice"],
        t
    )
)

print("\nSelected threshold:", best_threshold)
print(
    "Selected Dev defect-present Dice:",
    round(
        dev_results[str(best_threshold)]["mean_present_dice"],
        4
    )
)

print(
    "Selected Dev empty-image FP pixels:",
    round(
        dev_results[str(best_threshold)]["mean_empty_fp_pixels"]
    )
)


# Save threshold decision.
threshold_path = OUTPUT_DIR / "selected_threshold.json"

threshold_record = {
    "threshold": best_threshold,
    "selection_metric": "mean_present_dice",
    "selection_split": "dev",
    "thresholds_tested": THRESHOLDS,
    "dev_results": dev_results,
}

with open(threshold_path, "w", encoding="utf-8") as f:
    json.dump(threshold_record, f, indent=2)

print("Threshold decision saved:", threshold_path)


# ============================================================
# 8. FINAL TEST EVALUATION
# E is evaluated once with the locked threshold.
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")
print("Locked threshold:", best_threshold)
print("Test images:", len(TEST_E))

test_bucket = new_metric_bucket()

visual_examples = []
empty_examples = []
present_examples = []

with torch.no_grad():

    for index, image_id in enumerate(TEST_E, start=1):

        image, gt_mask = load_image_and_mask(image_id)

        probabilities = predict_full_width(image)
        pred_mask = probabilities >= best_threshold

        update_metrics(
            test_bucket,
            pred_mask,
            gt_mask
        )

        if len(visual_examples) < 6:

            if gt_mask.sum() == 0 and len(empty_examples) < 3:
                empty_examples.append(
                    (image_id, image, gt_mask, pred_mask)
                )

            elif gt_mask.sum() > 0 and len(present_examples) < 3:
                present_examples.append(
                    (image_id, image, gt_mask, pred_mask)
                )

        if index % 100 == 0:
            print(f"Processed Test images: {index}/{len(TEST_E)}")


test_metrics = finalize_metrics(test_bucket)

print("\nFINAL TEST RESULTS")
print("-" * 55)

for key, value in test_metrics.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# 9. SAVE TEST METRICS
# ============================================================

test_results_path = OUTPUT_DIR / "unet_test_metrics.json"

test_record = {
    "model": "U-Net",
    "checkpoint": str(CHECKPOINT_PATH),
    "best_training_epoch": checkpoint["epoch"],
    "threshold": best_threshold,
    "test_split": "test_E",
    "metrics": test_metrics,
}

with open(test_results_path, "w", encoding="utf-8") as f:
    json.dump(test_record, f, indent=2)

print("\nTest metrics saved:", test_results_path)


# ============================================================
# 10. SAVE VISUALIZATIONS
# ============================================================

visual_examples = empty_examples + present_examples

if visual_examples:

    fig, axes = plt.subplots(
        len(visual_examples),
        3,
        figsize=(16, 4 * len(visual_examples)),
        squeeze=False
    )

    for row, (image_id, image, gt_mask, pred_mask) in enumerate(
        visual_examples
    ):

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{image_id} - Original")

        axes[row, 1].imshow(gt_mask, cmap="gray")
        axes[row, 1].set_title("Ground Truth")

        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 2].set_title(
            f"Prediction (threshold={best_threshold})"
        )

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()

    visualization_path = OUTPUT_DIR / "unet_test_examples.png"

    plt.savefig(
        visualization_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Visualization saved:", visualization_path)


# ============================================================
# 11. FINISH
# ============================================================

print("\n" + "=" * 60)
print("U-NET EVALUATION COMPLETED")
print("=" * 60)

print("Selected threshold:", best_threshold)
print("Test Dice:", round(test_metrics["mean_present_dice"], 4))
print("Test IoU:", round(test_metrics["mean_present_iou"], 4))

print("\nNo model weights or dataset files were modified.")
