
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"E:\SAM")
DATA_DIR = PROJECT_DIR / "data"
MANIFEST_DIR = PROJECT_DIR / "manifests"

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
SPLIT_MANIFEST = MANIFEST_DIR / "split_manifest_v2.json"

EXPECTED_SHAPE = (256, 1600)
EXPECTED_SPLITS = {
    "S1": 50,
    "dev": 200,
    "test_E": 1200,
    "H": 10,
}


# ============================================================
# UTILITIES
# ============================================================

def check(condition, message):
    if not condition:
        raise RuntimeError(f"FAILED: {message}")
    print(f"[PASS] {message}")


def decode_rle(encoded_pixels, height=256, width=1600):
    values = np.asarray(
        [int(x) for x in encoded_pixels.split()],
        dtype=np.int64
    )

    if len(values) == 0 or len(values) % 2 != 0:
        raise ValueError("Invalid RLE sequence.")

    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths

    if np.any(starts < 0):
        raise ValueError("RLE contains negative starting positions.")

    if np.any(lengths <= 0):
        raise ValueError("RLE contains non-positive lengths.")

    if np.any(ends > height * width):
        raise ValueError("RLE exceeds image dimensions.")

    flat_mask = np.zeros(height * width, dtype=np.uint8)

    for start, end in zip(starts, ends):
        flat_mask[start:end] = 1

    return flat_mask.reshape((height, width), order="F")


# ============================================================
# 1. CHECK PROJECT PATHS
# ============================================================

print("\n========== 1. PROJECT PATHS ==========")

check(PROJECT_DIR.is_dir(), "Project directory exists")
check(TRAIN_CSV.is_file(), "Training CSV exists")
check(TRAIN_IMAGES_DIR.is_dir(), "Training images directory exists")
check(SPLIT_MANIFEST.is_file(), "Split manifest exists")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========== 2. LOAD DATA ==========")

df = pd.read_csv(TRAIN_CSV)

with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)

check(
    {"ImageId", "ClassId", "EncodedPixels"}.issubset(df.columns),
    "CSV contains required columns"
)

splits = manifest["splits"]

for split_name, expected_count in EXPECTED_SPLITS.items():
    check(split_name in splits, f"Split '{split_name}' exists")
    check(
        len(splits[split_name]) == expected_count,
        f"{split_name} contains {expected_count} images"
    )

print(f"\nCSV annotation rows: {len(df)}")
print(f"Unique annotated images: {df['ImageId'].nunique()}")



# ============================================================
# 3. CHECK SPLIT INTEGRITY
# ============================================================

print("\n========== 3. SPLIT INTEGRITY ==========")

split_sets = {
    name: set(splits[name])
    for name in EXPECTED_SPLITS
}

# These splits must be mutually exclusive.
independent_splits = ["S1", "dev", "test_E"]

for i, name_a in enumerate(independent_splits):
    for name_b in independent_splits[i + 1:]:
        overlap = split_sets[name_a] & split_sets[name_b]

        check(
            len(overlap) == 0,
            f"{name_a} and {name_b} have no overlapping images"
        )

# H is intentionally a subset of test_E.
h_outside_test = split_sets["H"] - split_sets["test_E"]

check(
    len(h_outside_test) == 0,
    "H is completely contained within test_E"
)

check(
    len(split_sets["H"] & split_sets["test_E"]) == 10,
    "All 10 human-evaluation images belong to test_E"
)

# Check total unique images across independent splits.
all_independent_ids = set.union(
    *(split_sets[name] for name in independent_splits)
)

check(
    len(all_independent_ids) == 50 + 200 + 1200,
    "S1, dev and test_E contain exactly 1450 unique images"
)


# ============================================================
# 4. CHECK IMAGE FILES AND DIMENSIONS
# ============================================================

print("\n========== 4. IMAGE VALIDATION ==========")

all_ids = sorted(all_independent_ids)

missing_images = [
    image_id
    for image_id in all_ids
    if not (TRAIN_IMAGES_DIR / image_id).is_file()
]

check(
    len(missing_images) == 0,
    f"All {len(all_ids)} split images exist"
)

sample_ids = (
    splits["S1"][:3]
    + splits["dev"][:3]
    + splits["test_E"][:3]
    + splits["H"][:1]
)

for image_id in sample_ids:
    image_path = TRAIN_IMAGES_DIR / image_id

    with Image.open(image_path) as img:
        check(
            img.size == (EXPECTED_SHAPE[1], EXPECTED_SHAPE[0]),
            f"{image_id} has dimensions 1600x256"
        )


# ============================================================
# 5. CHECK ANNOTATIONS AND RLE DECODING
# ============================================================

print("\n========== 5. ANNOTATION VALIDATION ==========")

rles_by_image = (
    df.groupby("ImageId")["EncodedPixels"]
    .apply(list)
    .to_dict()
)

print("Defect classes:")
for class_id, count in df["ClassId"].value_counts().sort_index().items():
    print(f"  Class {class_id}: {count} annotation rows")

check(
    set(df["ClassId"].unique()).issubset({1, 2, 3, 4}),
    "Only the four expected defect classes are present"
)

# Verify one mask per sampled image with annotations.
annotated_sample_ids = [
    image_id
    for image_id in sample_ids
    if image_id in rles_by_image
]

check(
    len(annotated_sample_ids) > 0,
    "Annotated sample images are available"
)

for image_id in annotated_sample_ids:
    combined_mask = np.zeros(EXPECTED_SHAPE, dtype=np.uint8)

    for rle in rles_by_image[image_id]:
        combined_mask |= decode_rle(rle)

    check(
        combined_mask.shape == EXPECTED_SHAPE,
        f"{image_id} mask dimensions are correct"
    )

    check(
        set(np.unique(combined_mask)).issubset({0, 1}),
        f"{image_id} mask is binary"
    )

    print(
        f"  {image_id}: "
        f"{int(combined_mask.sum())} defect pixels"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("DATA VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 55)

print(f"Total CSV rows: {len(df)}")
print(f"Total annotated images: {df['ImageId'].nunique()}")
print(f"Total split images: {len(all_ids)}")
print(f"Training images (S1): {len(splits['S1'])}")
print(f"Development images: {len(splits['dev'])}")
print(f"Test images: {len(splits['test_E'])}")
print(f"Human evaluation images: {len(splits['H'])}")

print("\nNo data or manifest files were modified.")
