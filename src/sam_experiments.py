
from pathlib import Path
from collections import defaultdict
import csv
import json
import time

import cv2
import numpy as np
import torch

from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"E:\SAM")

DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "train_images"
CSV_PATH = DATA_DIR / "train.csv"

MANIFEST_PATH = ROOT / "manifests" / "split_manifest_v2.json"
CHECKPOINT = ROOT / "checkpoints" / "sam" / "sam_vit_b_01ec64.pth"

OUTPUT_DIR = ROOT / "outputs" / "sam"
VIS_DIR = OUTPUT_DIR / "visualizations"

MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEIGHT = 256
WIDTH = 1600

# Fixed automatic mask generation settings.
# Keep these unchanged for both Dev and Test E.
AMG_POINTS_PER_SIDE = 16
AMG_PRED_IOU_THRESH = 0.86
AMG_STABILITY_THRESH = 0.92
AMG_MIN_REGION_AREA = 100

# Save a small number of visual examples per split.
VISUALS_PER_SPLIT = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RLE DECODING
# ============================================================

def decode_rle(encoded_pixels):
    """
    Decode the dataset's 1-indexed, column-major RLE.
    Returns a binary mask of shape (256, 1600).
    """

    mask = np.zeros(HEIGHT * WIDTH, dtype=np.uint8)

    if not encoded_pixels or not encoded_pixels.strip():
        return mask.reshape((HEIGHT, WIDTH), order="F")

    values = np.asarray(
        encoded_pixels.split(),
        dtype=np.int64
    )

    if len(values) % 2 != 0:
        raise ValueError("Invalid RLE: odd number of values")

    starts = values[0::2] - 1
    lengths = values[1::2]
    ends = starts + lengths

    if np.any(starts < 0) or np.any(ends > HEIGHT * WIDTH):
        raise ValueError("RLE values are outside image boundaries")

    for start, end in zip(starts, ends):
        mask[start:end] = 1

    return mask.reshape((HEIGHT, WIDTH), order="F")


# ============================================================
# LOAD DATA AND MANIFEST
# ============================================================

def load_manifest():
    with open(MANIFEST_PATH, "r") as file:
        manifest = json.load(file)

    splits = manifest["splits"]

    for name in ("dev", "test_E"):
        if name not in splits:
            raise KeyError(f"Missing split in manifest: {name}")

    return splits


def load_ground_truth():
    """
    Combine all class annotations into one binary union mask
    for each annotated image.
    """

    masks = {}

    print("\nLoading and decoding ground-truth masks...")

    with open(CSV_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)

        required = {"ImageId", "ClassId", "EncodedPixels"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns: {required}"
            )

        for row in reader:
            image_id = row["ImageId"]

            if image_id not in masks:
                masks[image_id] = np.zeros(
                    (HEIGHT, WIDTH),
                    dtype=np.uint8
                )

            decoded = decode_rle(row["EncodedPixels"])
            masks[image_id] |= decoded

    print(f"Annotated images loaded: {len(masks)}")

    return masks


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, ground_truth):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = int(np.logical_and(prediction, ground_truth).sum())
    fp = int(np.logical_and(prediction, ~ground_truth).sum())
    fn = int(np.logical_and(~prediction, ground_truth).sum())

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if (2 * tp + fp + fn) > 0
        else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_pixels": int(prediction.sum()),
        "ground_truth_pixels": int(ground_truth.sum()),
    }


def summarize_results(records):
    if not records:
        return {"image_count": 0}

    present = [r for r in records if r["ground_truth_pixels"] > 0]
    empty = [r for r in records if r["ground_truth_pixels"] == 0]

    total_tp = sum(r["tp"] for r in records)
    total_fp = sum(r["fp"] for r in records)
    total_fn = sum(r["fn"] for r in records)

    total_present_tp = sum(r["tp"] for r in present)
    total_present_fp = sum(r["fp"] for r in present)
    total_present_fn = sum(r["fn"] for r in present)

    result = {
        "image_count": len(records),
        "defect_present_count": len(present),
        "defect_free_count": len(empty),
        "mean_dice_all_images": float(
            np.mean([r["dice"] for r in records])
        ),
        "mean_iou_all_images": float(
            np.mean([r["iou"] for r in records])
        ),
        "global_pixel_dice": float(
            2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1)
        ),
        "global_pixel_iou": float(
            total_tp / max(total_tp + total_fp + total_fn, 1)
        ),
        "global_precision": float(
            total_tp / max(total_tp + total_fp, 1)
        ),
        "global_recall": float(
            total_tp / max(total_tp + total_fn, 1)
        ),
    }

    if present:
        result.update({
            "mean_present_dice": float(
                np.mean([r["dice"] for r in present])
            ),
            "mean_present_iou": float(
                np.mean([r["iou"] for r in present])
            ),
            "present_precision": float(
                total_present_tp /
                max(total_present_tp + total_present_fp, 1)
            ),
            "present_recall": float(
                total_present_tp /
                max(total_present_tp + total_present_fn, 1)
            ),
        })

    if empty:
        result.update({
            "mean_empty_fp_pixels": float(
                np.mean([r["fp"] for r in empty])
            ),
            "empty_image_false_positive_rate": float(
                np.mean([r["fp"] > 0 for r in empty])
            ),
        })

    return result


# ============================================================
# ORACLE PROMPT GENERATION
# ============================================================

def make_point_prompt(ground_truth):
    """
    Select the deepest pixel of the largest connected defect
    component. This is a deterministic positive oracle point.
    """

    binary = ground_truth.astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    if count <= 1:
        return None

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8)

    distance = cv2.distanceTransform(
        largest_component,
        cv2.DIST_L2,
        5
    )

    y, x = np.unravel_index(np.argmax(distance), distance.shape)

    return np.array([[x, y]], dtype=np.float32)


def make_box_prompt(ground_truth):
    """
    Bounding box around the complete binary defect union.
    """

    ys, xs = np.where(ground_truth > 0)

    if len(xs) == 0:
        return None

    # SAM expects XYXY coordinates.
    return np.array(
        [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]],
        dtype=np.float32
    )


# ============================================================
# SAM PREDICTION METHODS
# ============================================================

def select_best_mask(masks, scores):
    best_index = int(np.argmax(scores))
    return masks[best_index].astype(np.uint8)


def predict_point(predictor, ground_truth):
    point = make_point_prompt(ground_truth)

    if point is None:
        raise ValueError("Cannot create point prompt for empty mask")

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )

    return select_best_mask(masks, scores)


def predict_box(predictor, ground_truth):
    box = make_box_prompt(ground_truth)

    if box is None:
        raise ValueError("Cannot create box prompt for empty mask")

    masks, scores, _ = predictor.predict(
        box=box[0],
        multimask_output=False,
    )

    return select_best_mask(masks, scores)


def predict_automatic(generator, image):
    generated_masks = generator.generate(image)

    # Treat the union of generated SAM masks as the binary
    # defect prediction. No semantic classifier is applied.
    prediction = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8
    )

    for item in generated_masks:
        prediction |= item["segmentation"].astype(np.uint8)

    return prediction, len(generated_masks)


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    image,
    ground_truth,
    prediction,
    output_path,
    title
):
    image_copy = image.copy()

    overlay = image_copy.copy()
    overlay[ground_truth > 0] = (
        0.5 * overlay[ground_truth > 0]
        + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    overlay[prediction > 0] = (
        0.5 * overlay[prediction > 0]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Green = ground truth; red = prediction.
    # Overlap appears as a mixture.
    cv2.putText(
        overlay,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_split(
    split_name,
    image_ids,
    ground_truth_masks,
    predictor,
    generator,
):
    print("\n" + "=" * 65)
    print(f"EVALUATING SPLIT: {split_name}")
    print(f"Images: {len(image_ids)}")
    print("=" * 65)

    methods = {
        "point_oracle": [],
        "box_oracle": [],
        "automatic": [],
    }

    timings = defaultdict(list)
    automatic_mask_counts = []

    visual_count = 0
    split_visual_dir = VIS_DIR / split_name
    split_visual_dir.mkdir(parents=True, exist_ok=True)

    for index, image_id in enumerate(image_ids, start=1):

        image_path = IMAGE_DIR / image_id

        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        if image.shape[:2] != (HEIGHT, WIDTH):
            raise ValueError(
                f"Unexpected image shape for {image_id}: {image.shape}"
            )

        ground_truth = ground_truth_masks.get(
            image_id,
            np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        )

        is_present = bool(ground_truth.any())

        # Compute image embedding once; reuse for point and box.
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        predictor.set_image(image)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        embedding_time = time.perf_counter() - start

        # ----------------------------------------------------
        # POINT ORACLE
        # ----------------------------------------------------

        if is_present:
            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            point_prediction = predict_point(
                predictor,
                ground_truth
            )

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            point_time = time.perf_counter() - start

            point_metrics = calculate_metrics(
                point_prediction,
                ground_truth
            )

            point_metrics.update({
                "image_id": image_id,
                "embedding_time_seconds": embedding_time,
                "prompt_inference_time_seconds": point_time,
                "total_time_seconds": embedding_time + point_time,
                "ground_truth_pixels": int(ground_truth.sum()),
            })

            methods["point_oracle"].append(point_metrics)

        # ----------------------------------------------------
        # BOX ORACLE
        # ----------------------------------------------------

        if is_present:
            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            box_prediction = predict_box(
                predictor,
                ground_truth
            )

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            box_time = time.perf_counter() - start

            box_metrics = calculate_metrics(
                box_prediction,
                ground_truth
            )

            box_metrics.update({
                "image_id": image_id,
                "embedding_time_seconds": embedding_time,
                "prompt_inference_time_seconds": box_time,
                "total_time_seconds": embedding_time + box_time,
                "ground_truth_pixels": int(ground_truth.sum()),
            })

            methods["box_oracle"].append(box_metrics)

        # ----------------------------------------------------
        # AUTOMATIC MASK GENERATION
        # ----------------------------------------------------

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        automatic_prediction, mask_count = predict_automatic(
            generator,
            image
        )

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        automatic_time = time.perf_counter() - start

        automatic_metrics = calculate_metrics(
            automatic_prediction,
            ground_truth
        )

        automatic_metrics.update({
            "image_id": image_id,
            "inference_time_seconds": automatic_time,
            "generated_mask_count": mask_count,
            "ground_truth_pixels": int(ground_truth.sum()),
        })

        methods["automatic"].append(automatic_metrics)

        timings["embedding"].append(embedding_time)
        timings["automatic"].append(automatic_time)

        if is_present:
            timings["point_prompt"].append(point_time)
            timings["box_prompt"].append(box_time)

        automatic_mask_counts.append(mask_count)

        # ----------------------------------------------------
        # SAVE A FEW VISUAL EXAMPLES
        # ----------------------------------------------------

        if visual_count < VISUALS_PER_SPLIT and (
            is_present or visual_count == 0
        ):
            save_visualization(
                image,
                ground_truth,
                automatic_prediction,
                split_visual_dir / f"{visual_count + 1}_automatic.png",
                f"Automatic | {image_id}"
            )

            if is_present:
                save_visualization(
                    image,
                    ground_truth,
                    point_prediction,
                    split_visual_dir / f"{visual_count + 1}_point.png",
                    f"Point oracle | {image_id}"
                )

                save_visualization(
                    image,
                    ground_truth,
                    box_prediction,
                    split_visual_dir / f"{visual_count + 1}_box.png",
                    f"Box oracle | {image_id}"
                )

            visual_count += 1

        if index % 25 == 0 or index == len(image_ids):
            print(
                f"[{index}/{len(image_ids)}] "
                f"Processed: {image_id} | "
                f"Generated masks: {mask_count} | "
                f"Automatic time: {automatic_time:.2f}s"
            )

    # --------------------------------------------------------
    # SUMMARIZE
    # --------------------------------------------------------

    split_result = {
        "split": split_name,
        "image_count": len(image_ids),
        "protocol_notes": {
            "point_and_box": (
                "Evaluated only on defect-present images. "
                "Prompts are derived from ground-truth masks."
            ),
            "automatic": (
                "Evaluated on all images, including defect-free images."
            ),
        },
        "methods": {},
        "timing": {},
        "automatic_mask_statistics": {},
    }

    for method_name, records in methods.items():
        split_result["methods"][method_name] = {
            "summary": summarize_results(records),
            "per_image": records,
        }

    for timing_name, values in timings.items():
        split_result["timing"][timing_name] = {
            "mean_seconds": float(np.mean(values)) if values else None,
            "median_seconds": float(np.median(values)) if values else None,
        }

    split_result["automatic_mask_statistics"] = {
        "mean_generated_masks": float(
            np.mean(automatic_mask_counts)
        ),
        "median_generated_masks": float(
            np.median(automatic_mask_counts)
        ),
    }

    return split_result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("FROZEN SAM STEEL DEFECT EXPERIMENTS")
    print("=" * 65)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not CSV_PATH.exists():
        raise FileNotFoundError(CSV_PATH)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    print(f"Device: {DEVICE}")

    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits = load_manifest()
    ground_truth_masks = load_ground_truth()

    # Verify expected split sizes before starting.
    assert len(splits["dev"]) == 200, (
        f"Expected 200 Dev images, got {len(splits['dev'])}"
    )

    assert len(splits["test_E"]) == 1200, (
        f"Expected 1200 Test E images, got {len(splits['test_E'])}"
    )

    # --------------------------------------------------------
    # LOAD FROZEN SAM
    # --------------------------------------------------------

    print("\nLoading frozen SAM ViT-B...")

    sam = sam_model_registry[MODEL_TYPE](
        checkpoint=str(CHECKPOINT)
    )

    sam.to(device=DEVICE)
    sam.eval()

    for parameter in sam.parameters():
        parameter.requires_grad = False

    trainable_parameters = sum(
        p.numel() for p in sam.parameters() if p.requires_grad
    )

    if trainable_parameters != 0:
        raise RuntimeError("SAM is not fully frozen")

    predictor = SamPredictor(sam)

    # --------------------------------------------------------
    # AUTOMATIC MASK GENERATOR
    # --------------------------------------------------------

    print("Initializing automatic mask generator...")

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=AMG_POINTS_PER_SIDE,
        pred_iou_thresh=AMG_PRED_IOU_THRESH,
        stability_score_thresh=AMG_STABILITY_THRESH,
        crop_n_layers=0,
        min_mask_region_area=AMG_MIN_REGION_AREA,
    )

    print("SAM ready. Starting experiments.")

    # --------------------------------------------------------
    # DEV EVALUATION
    # --------------------------------------------------------

    dev_results = evaluate_split(
        "dev",
        splits["dev"],
        ground_truth_masks,
        predictor,
        generator,
    )

    dev_path = OUTPUT_DIR / "sam_dev_results.json"

    with open(dev_path, "w") as file:
        json.dump(dev_results, file, indent=2)

    print(f"\nDev results saved: {dev_path}")

    # --------------------------------------------------------
    # TEST E EVALUATION
    # Fixed configuration; no tuning on Test E.
    # --------------------------------------------------------

    test_results = evaluate_split(
        "test_E",
        splits["test_E"],
        ground_truth_masks,
        predictor,
        generator,
    )

    test_path = OUTPUT_DIR / "sam_test_results.json"

    with open(test_path, "w") as file:
        json.dump(test_results, file, indent=2)

    print(f"\nTest E results saved: {test_path}")

    # --------------------------------------------------------
    # SAVE EXPERIMENT CONFIGURATION
    # --------------------------------------------------------

    config = {
        "model": "SAM ViT-B",
        "model_type": MODEL_TYPE,
        "frozen": True,
        "trainable_parameters": trainable_parameters,
        "device": DEVICE,
        "automatic_mask_generator": {
            "points_per_side": AMG_POINTS_PER_SIDE,
            "pred_iou_thresh": AMG_PRED_IOU_THRESH,
            "stability_score_thresh": AMG_STABILITY_THRESH,
            "crop_n_layers": 0,
            "min_mask_region_area": AMG_MIN_REGION_AREA,
        },
        "prompt_protocol": {
            "point": "Deepest pixel of largest GT connected component",
            "box": "Bounding box around complete binary GT union",
            "point_and_box_are_oracle_prompts": True,
            "empty_images_in_point_box_evaluation": False,
            "empty_images_in_automatic_evaluation": True,
        },
        "splits": {
            "dev": len(splits["dev"]),
            "test_E": len(splits["test_E"]),
        },
    }

    config_path = OUTPUT_DIR / "sam_experiment_config.json"

    with open(config_path, "w") as file:
        json.dump(config, file, indent=2)

    print("\n" + "=" * 65)
    print("SAM EXPERIMENTS COMPLETED")
    print("=" * 65)
    print(f"Dev results:    {dev_path}")
    print(f"Test results:   {test_path}")
    print(f"Config:         {config_path}")
    print(f"Visualizations: {VIS_DIR}")


if __name__ == "__main__":
    main()
