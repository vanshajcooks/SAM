
from pathlib import Path
import json
import time

import torch
import torchvision
import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(r"E:\SAM")
CHECKPOINT = ROOT / "checkpoints" / "sam" / "sam_vit_b_01ec64.pth"
OUTPUT_DIR = ROOT / "outputs" / "sam"

MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# ENVIRONMENT CHECK
# --------------------------------------------------

print("=" * 60)
print("SAM ENVIRONMENT SETUP")
print("=" * 60)

print(f"PyTorch:       {torch.__version__}")
print(f"Torchvision:   {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device:        {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU:           {torch.cuda.get_device_name(0)}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"SAM checkpoint not found: {CHECKPOINT}"
    )

print(f"\nCheckpoint:    {CHECKPOINT}")
print(f"Checkpoint size: {CHECKPOINT.stat().st_size / (1024**3):.2f} GB")


# --------------------------------------------------
# LOAD FROZEN SAM
# --------------------------------------------------

print("\nLoading SAM ViT-B...")

start_time = time.time()

sam = sam_model_registry[MODEL_TYPE](
    checkpoint=str(CHECKPOINT)
)

sam.to(device=DEVICE)
sam.eval()

# Freeze all parameters: no fine-tuning
for parameter in sam.parameters():
    parameter.requires_grad = False

load_time = time.time() - start_time

total_parameters = sum(
    parameter.numel() for parameter in sam.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in sam.parameters()
    if parameter.requires_grad
)

print(f"SAM loaded in {load_time:.2f} seconds")
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")


# --------------------------------------------------
# INFERENCE SMOKE TEST
# --------------------------------------------------

print("\nRunning inference smoke test...")

# Use an actual training image
image_dir = ROOT / "data" / "train_images"

image_files = sorted(image_dir.glob("*"))

if not image_files:
    raise FileNotFoundError(
        f"No images found in {image_dir}"
    )

image_path = image_files[0]

image_bgr = cv2.imread(str(image_path))

if image_bgr is None:
    raise RuntimeError(f"Could not read image: {image_path}")

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

predictor = SamPredictor(sam)

start_time = time.time()

predictor.set_image(image_rgb)

embedding_time = time.time() - start_time

print(f"Test image: {image_path.name}")
print(f"Image shape: {image_rgb.shape}")
print(f"Image embedding time: {embedding_time:.3f} seconds")
print(f"Embedding shape: {predictor.get_image_embedding().shape}")


# --------------------------------------------------
# SAVE SETUP REPORT
# --------------------------------------------------

report = {
    "model": "SAM ViT-B",
    "model_type": MODEL_TYPE,
    "checkpoint": str(CHECKPOINT),
    "device": DEVICE,
    "gpu": (
        torch.cuda.get_device_name(0)
        if DEVICE == "cuda"
        else None
    ),
    "torch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
    "frozen": trainable_parameters == 0,
    "load_time_seconds": round(load_time, 3),
    "test_image": image_path.name,
    "test_image_shape": list(image_rgb.shape),
    "embedding_shape": list(
        predictor.get_image_embedding().shape
    ),
    "embedding_time_seconds": round(embedding_time, 3),
}

report_path = OUTPUT_DIR / "sam_setup_report.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\n" + "=" * 60)
print("SAM SETUP COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Report saved: {report_path}")
