
"""
02_unet_train.py

Train a U-Net for binary steel defect segmentation.

Training:
    S1: 50 images

Validation:
    Dev: 200 images

Input:
    RGB image crops of shape 256 x 512

Output:
    Binary defect segmentation mask

Checkpoint selection:
    Mean Dice on defect-present development images.

Also reports:
    False-positive pixels on defect-free development images.
"""

from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\SAM")

DATA_DIR = PROJECT_DIR / "data"
MANIFEST_DIR = PROJECT_DIR / "manifests"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "unet"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
SPLIT_MANIFEST = MANIFEST_DIR / "split_manifest_v2.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 1600

CROP_HEIGHT = 256
CROP_WIDTH = 512

BATCH_SIZE = 2
EPOCHS = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

THRESHOLD = 0.5
RANDOM_SEED = 532

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=" * 60)
print("U-NET STEEL DEFECT SEGMENTATION")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. LOAD DATA AND SPLITS
# ============================================================

print("\n========== LOADING DATA ==========")

df = pd.read_csv(TRAIN_CSV)

with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)

S1 = list(manifest["splits"]["S1"])
DEV = list(manifest["splits"]["dev"])

assert len(S1) == 50, "Unexpected S1 size"
assert len(DEV) == 200, "Unexpected Dev size"

print("Training images:", len(S1))
print("Development images:", len(DEV))

rles_by_image = (
    df.groupby("ImageId")["EncodedPixels"]
    .apply(list)
    .to_dict()
)


# ============================================================
# 4. RLE DECODING AND IMAGE LOADING
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
        or np.any(ends > IMAGE_HEIGHT * IMAGE_WIDTH)
    ):
        raise ValueError("RLE values exceed image boundaries")

    flat = np.zeros(
        IMAGE_HEIGHT * IMAGE_WIDTH,
        dtype=np.uint8
    )

    for start, end in zip(starts, ends):
        flat[start:end] = 1

    return flat.reshape(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        order="F"
    )


def load_image_and_mask(image_id):

    image_path = TRAIN_IMAGES_DIR / image_id

    with Image.open(image_path) as img:
        image = np.asarray(
            img.convert("RGB"),
            dtype=np.uint8
        )

    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(
            f"Unexpected image shape for {image_id}: {image.shape}"
        )

    mask = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.uint8
    )

    for rle in rles_by_image.get(image_id, []):
        mask |= decode_rle(rle)

    return image, mask


# ============================================================
# 5. TRAINING DATASET
# ============================================================

class SteelDefectDataset(Dataset):

    def __init__(self, image_ids):
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image, mask = load_image_and_mask(image_id)

        max_left = IMAGE_WIDTH - CROP_WIDTH

        # Attempt a balanced crop selection:
        # 50% defect-containing crops
        # 50% defect-free crops

        want_defect = random.random() < 0.5

        selected_left = None

        for _ in range(20):

            left = random.randint(0, max_left)

            candidate_mask = mask[
                :,
                left:left + CROP_WIDTH
            ]

            has_defect = bool(candidate_mask.any())

            if has_defect == want_defect:
                selected_left = left
                break

        # Fallback if the requested crop type was unavailable.
        if selected_left is None:
            selected_left = random.randint(0, max_left)

        image = image[
            :,
            selected_left:selected_left + CROP_WIDTH,
            :
        ]

        mask = mask[
            :,
            selected_left:selected_left + CROP_WIDTH
        ]

        image_tensor = (
            torch.from_numpy(
                image.astype(np.float32).copy() / 255.0
            )
            .permute(2, 0, 1)
        )

        mask_tensor = torch.from_numpy(
            mask.astype(np.float32).copy()
        ).unsqueeze(0)

        # Horizontal flip augmentation.
        if random.random() < 0.5:

            image_tensor = torch.flip(
                image_tensor,
                dims=[2]
            )

            mask_tensor = torch.flip(
                mask_tensor,
                dims=[2]
            )

        return image_tensor, mask_tensor


# ============================================================
# 6. FULL-WIDTH VALIDATION
# ============================================================

# Four overlapping crops cover the complete 1600-pixel width.
CROP_STARTS = (0, 363, 726, 1088)


def evaluate_full_width(model, image_ids):

    model.eval()

    positive_dice = []
    positive_iou = []

    empty_fp_pixels = []
    empty_fp_image_rate = []

    all_dice = []
    all_iou = []

    with torch.no_grad():

        for image_id in image_ids:

            image, gt_mask = load_image_and_mask(image_id)

            probability_sum = np.zeros(
                (IMAGE_HEIGHT, IMAGE_WIDTH),
                dtype=np.float32
            )

            count_sum = np.zeros(
                (IMAGE_HEIGHT, IMAGE_WIDTH),
                dtype=np.float32
            )

            for left in CROP_STARTS:

                crop = image[
                    :,
                    left:left + CROP_WIDTH,
                    :
                ]

                x = (
                    torch.from_numpy(
                        crop.astype(np.float32).copy() / 255.0
                    )
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                logits = model(x)

                probabilities = (
                    torch.sigmoid(logits)[0, 0]
                    .cpu()
                    .numpy()
                )

                probability_sum[
                    :,
                    left:left + CROP_WIDTH
                ] += probabilities

                count_sum[
                    :,
                    left:left + CROP_WIDTH
                ] += 1

            probabilities = (
                probability_sum / np.maximum(count_sum, 1)
            )

            pred_mask = probabilities >= THRESHOLD

            intersection = np.logical_and(
                pred_mask,
                gt_mask
            ).sum()

            pred_pixels = int(pred_mask.sum())
            gt_pixels = int(gt_mask.sum())

            union = np.logical_or(
                pred_mask,
                gt_mask
            ).sum()

            dice = (
                2.0 * intersection / (pred_pixels + gt_pixels)
                if pred_pixels + gt_pixels > 0
                else 1.0
            )

            iou = (
                intersection / union
                if union > 0
                else 1.0
            )

            all_dice.append(float(dice))
            all_iou.append(float(iou))

            if gt_pixels > 0:

                positive_dice.append(float(dice))
                positive_iou.append(float(iou))

            else:

                empty_fp_pixels.append(pred_pixels)

                empty_fp_image_rate.append(
                    float(pred_pixels > 0)
                )

    if not positive_dice:
        raise RuntimeError(
            "No defect-present images found in development set."
        )

    return {
        "mean_present_dice": float(np.mean(positive_dice)),
        "mean_present_iou": float(np.mean(positive_iou)),
        "mean_empty_fp_pixels": (
            float(np.mean(empty_fp_pixels))
            if empty_fp_pixels else 0.0
        ),
        "empty_image_false_positive_rate": (
            float(np.mean(empty_fp_image_rate))
            if empty_fp_image_rate else 0.0
        ),
        "mean_all_image_dice": float(np.mean(all_dice)),
        "mean_all_image_iou": float(np.mean(all_iou)),
        "present_images": len(positive_dice),
        "empty_images": len(empty_fp_pixels),
    }


# ============================================================
# 7. U-NET ARCHITECTURE
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
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

        self.up4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )

        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(
            32, 1, kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 8. LOSS FUNCTIONS
# ============================================================

bce_loss_fn = nn.BCEWithLogitsLoss()


def soft_dice_loss(logits, targets, eps=1e-6):

    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        (2.0 * intersection + eps)
        / (denominator + eps)
    )

    return 1.0 - dice.mean()


def combined_loss(logits, targets):

    bce = bce_loss_fn(logits, targets)
    dice = soft_dice_loss(logits, targets)

    return bce + dice


# ============================================================
# 9. CREATE DATA LOADERS
# ============================================================

print("\n========== CREATING DATA LOADERS ==========")

train_dataset = SteelDefectDataset(S1)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("Training samples:", len(train_dataset))
print("Training batches per epoch:", len(train_loader))


# ============================================================
# 10. INITIALIZE MODEL
# ============================================================

print("\n========== INITIALIZING MODEL ==========")

model = UNet().to(DEVICE)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("Model parameters:", f"{parameter_count:,}")

# Verify expected input/output dimensions.
test_input = torch.randn(
    1, 3, CROP_HEIGHT, CROP_WIDTH
).to(DEVICE)

with torch.no_grad():
    test_output = model(test_input)

assert test_output.shape == (
    1, 1, CROP_HEIGHT, CROP_WIDTH
)

print("Model shape test: PASS")

del test_input, test_output


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 11. TRAINING LOOP
# ============================================================

print("\n========== STARTING TRAINING ==========")

history = []

best_dev_dice = -1.0

checkpoint_path = (
    CHECKPOINT_DIR / "unet_best.pth"
)

history_path = (
    OUTPUT_DIR / "unet_training_history.json"
)

start_time = time.time()

for epoch in range(EPOCHS):

    model.train()

    train_losses = []

    epoch_start = time.time()

    for images, masks in train_loader:

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = combined_loss(logits, masks)

        loss.backward()

        optimizer.step()

        train_losses.append(loss.item())

    mean_train_loss = float(np.mean(train_losses))

    # Evaluate on all 200 development images.
    dev_metrics = evaluate_full_width(model, DEV)

    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": mean_train_loss,
        **dev_metrics
    }

    history.append(epoch_record)

    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {mean_train_loss:.4f} | "
        f"Present Dice: {dev_metrics['mean_present_dice']:.4f} | "
        f"Present IoU: {dev_metrics['mean_present_iou']:.4f} | "
        f"Empty FP Pixels: "
        f"{dev_metrics['mean_empty_fp_pixels']:.0f} | "
        f"Time: {epoch_time:.1f}s"
    )

    # Select checkpoint using defect-present Dice.
    if dev_metrics["mean_present_dice"] > best_dev_dice:

        best_dev_dice = dev_metrics["mean_present_dice"]

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_present_dice": best_dev_dice,
                "threshold": THRESHOLD,
                "seed": RANDOM_SEED,
            },
            checkpoint_path
        )

        print("  -> Best checkpoint saved.")


# ============================================================
# 12. SAVE TRAINING HISTORY
# ============================================================

with open(history_path, "w", encoding="utf-8") as f:

    json.dump(
        history,
        f,
        indent=2
    )


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

total_time = time.time() - start_time

best_epoch = max(
    history,
    key=lambda x: x["mean_present_dice"]
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"Total training time: {total_time / 60:.2f} minutes")

print("Best epoch:", best_epoch["epoch"])

print(
    "Best defect-present Dice:",
    round(best_epoch["mean_present_dice"], 4)
)

print(
    "Best defect-present IoU:",
    round(best_epoch["mean_present_iou"], 4)
)

print(
    "Empty-image mean false-positive pixels:",
    round(best_epoch["mean_empty_fp_pixels"])
)

print(
    "Empty-image false-positive rate:",
    round(best_epoch["empty_image_false_positive_rate"], 4)
)

print("\nCheckpoint:", checkpoint_path)
print("History:", history_path)

print("\nU-NET TRAINING SCRIPT FINISHED SUCCESSFULLY.")
