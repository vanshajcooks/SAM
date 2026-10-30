
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"E:\SAM")

SAM_DIR = ROOT / "outputs" / "sam"
UNET_DIR = ROOT / "outputs" / "unet"

OUTPUT_DIR = ROOT / "outputs" / "comparison"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")

    with open(path, "r") as file:
        return json.load(file)


sam_dev = load_json(SAM_DIR / "sam_dev_results.json")
sam_test = load_json(SAM_DIR / "sam_test_results.json")
unet_test = load_json(UNET_DIR / "unet_test_metrics.json")

unet_metrics = unet_test["metrics"]


# ============================================================
# EXTRACT TEST METRICS
# ============================================================

methods = [
    {
        "method": "U-Net (50 images)",
        "type": "Supervised",
        "scope": "All 1200 Test E images",
        "dice": unet_metrics["mean_present_dice"],
        "iou": unet_metrics["mean_present_iou"],
        "precision": unet_metrics["precision"],
        "recall": unet_metrics["recall"],
        "empty_fp": unet_metrics["mean_empty_fp_pixels"],
        "empty_fpr": unet_metrics["empty_image_false_positive_rate"],
        "inference_time": None,
    }
]

sam_method_names = {
    "point_oracle": "SAM Point (Oracle)",
    "box_oracle": "SAM Box (Oracle)",
    "automatic": "SAM Automatic",
}

for key, display_name in sam_method_names.items():

    summary = sam_test["methods"][key]["summary"]

    if key == "automatic":
        scope = "All 1200 Test E images"
        dice = summary["mean_present_dice"]
        iou = summary["mean_present_iou"]
        precision = summary["present_precision"]
        recall = summary["present_recall"]
        empty_fp = summary["mean_empty_fp_pixels"]
        empty_fpr = summary["empty_image_false_positive_rate"]

        inference_time = sam_test["timing"]["automatic"]["mean_seconds"]

    else:
        scope = "1100 defect-present images"
        dice = summary["mean_present_dice"]
        iou = summary["mean_present_iou"]
        precision = summary["present_precision"]
        recall = summary["present_recall"]
        empty_fp = None
        empty_fpr = None

        embedding_time = sam_test["timing"]["embedding"]["mean_seconds"]

        prompt_key = (
            "point_prompt"
            if key == "point_oracle"
            else "box_prompt"
        )

        prompt_time = sam_test["timing"][prompt_key]["mean_seconds"]

        # Image embedding + prompt inference.
        inference_time = embedding_time + prompt_time

    methods.append({
        "method": display_name,
        "type": "Frozen SAM",
        "scope": scope,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "empty_fp": empty_fp,
        "empty_fpr": empty_fpr,
        "inference_time": inference_time,
    })


# ============================================================
# SAVE CONSOLIDATED CSV
# ============================================================

csv_path = OUTPUT_DIR / "final_comparison.csv"

columns = [
    "method",
    "type",
    "scope",
    "dice",
    "iou",
    "precision",
    "recall",
    "empty_fp",
    "empty_fpr",
    "inference_time",
]

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for method in methods:
        writer.writerow(method)

print(f"Comparison CSV saved: {csv_path}")


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 105)
print("FINAL TEST E COMPARISON")
print("=" * 105)

print(
    f"{'Method':<24}"
    f"{'Dice':>12}"
    f"{'IoU':>12}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
    f"{'Time (s)':>14}"
)

print("-" * 105)

for method in methods:
    time_value = method["inference_time"]

    time_text = (
        f"{time_value:.4f}"
        if time_value is not None
        else "N/A"
    )

    print(
        f"{method['method']:<24}"
        f"{method['dice']:>12.4f}"
        f"{method['iou']:>12.4f}"
        f"{method['precision']:>14.4f}"
        f"{method['recall']:>12.4f}"
        f"{time_text:>14}"
    )


# ============================================================
# CHART 1: DICE COMPARISON
# ============================================================

labels = [m["method"] for m in methods]
dice_values = [m["dice"] for m in methods]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, dice_values)

plt.title("Test E: Mean Defect-Present Dice Comparison")
plt.ylabel("Mean Dice")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")

for bar, value in zip(bars, dice_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "dice_comparison.png",
    dpi=200
)
plt.close()


# ============================================================
# CHART 2: IOU COMPARISON
# ============================================================

iou_values = [m["iou"] for m in methods]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, iou_values)

plt.title("Test E: Mean Defect-Present IoU Comparison")
plt.ylabel("Mean IoU")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")

for bar, value in zip(bars, iou_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "iou_comparison.png",
    dpi=200
)
plt.close()


# ============================================================
# CHART 3: FALSE POSITIVES ON EMPTY IMAGES
# ============================================================

empty_methods = [
    m for m in methods
    if m["empty_fp"] is not None
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    [m["method"] for m in empty_methods],
    [m["empty_fp"] for m in empty_methods]
)

plt.title("Mean False-Positive Pixels on Defect-Free Test Images")
plt.ylabel("Mean False-Positive Pixels per Image")
plt.xticks(rotation=10, ha="right")

for bar, method in zip(bars, empty_methods):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 3000,
        f"{method['empty_fp']:,.0f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "empty_image_false_positives.png",
    dpi=200
)
plt.close()


# ============================================================
# CHART 4: SAM INFERENCE TIME
# ============================================================

sam_timing = [
    ("Point oracle", sam_test["timing"]["embedding"]["mean_seconds"]
     + sam_test["timing"]["point_prompt"]["mean_seconds"]),
    ("Box oracle", sam_test["timing"]["embedding"]["mean_seconds"]
     + sam_test["timing"]["box_prompt"]["mean_seconds"]),
    ("Automatic", sam_test["timing"]["automatic"]["mean_seconds"]),
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    [item[0] for item in sam_timing],
    [item[1] for item in sam_timing]
)

plt.title("SAM Mean Inference Time per Image")
plt.ylabel("Seconds")
plt.xticks(rotation=10)

for bar, item in zip(bars, sam_timing):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{item[1]:.3f}s",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sam_inference_time.png",
    dpi=200
)
plt.close()


# ============================================================
# DEV VS TEST E SUMMARY
# ============================================================

dev_test_lines = []

for key, display_name in sam_method_names.items():

    dev_summary = sam_dev["methods"][key]["summary"]
    test_summary = sam_test["methods"][key]["summary"]

    dev_dice = dev_summary["mean_present_dice"]
    test_dice = test_summary["mean_present_dice"]

    dev_test_lines.append(
        f"| {display_name} | {dev_dice:.4f} | {test_dice:.4f} |"
    )


# ============================================================
# GENERATE RESEARCH REPORT
# ============================================================

best_method = max(
    methods,
    key=lambda item: item["dice"]
)

unet_dice = unet_metrics["mean_present_dice"]
box_dice = sam_test["methods"]["box_oracle"]["summary"]["mean_present_dice"]
point_dice = sam_test["methods"]["point_oracle"]["summary"]["mean_present_dice"]
auto_dice = sam_test["methods"]["automatic"]["summary"]["mean_present_dice"]

report = f"""# Final Experimental Results

## Project

**Prompt or Label? A Cost-Aware Comparison of Frozen Segment Anything
(Point, Box and Automatic Prompting) and a 50-Image U-Net for Steel
Surface Defect Segmentation**

## 1. Experimental Setup

- Dataset: Steel surface defect dataset
- Image dimensions: 256 × 1600
- Defect classes: Four classes combined into a binary union mask
- U-Net training set: 50 images
- Development set: 200 images
- Final test set: 1,200 images
- Test composition: 1,100 defect-present and 100 defect-free images
- SAM: Original frozen Meta SAM ViT-B
- SAM fine-tuning: None

## 2. Test E Results

| Method | Evaluation scope | Mean defect-present Dice | Mean defect-present IoU |
|---|---|---:|---:|
| U-Net (50 images) | 1,100 defect-present images | {unet_dice:.4f} | {unet_metrics['mean_present_iou']:.4f} |
| SAM Point (Oracle) | 1,100 defect-present images | {point_dice:.4f} | {sam_test['methods']['point_oracle']['summary']['mean_present_iou']:.4f} |
| SAM Box (Oracle) | 1,100 defect-present images | {box_dice:.4f} | {sam_test['methods']['box_oracle']['summary']['mean_present_iou']:.4f} |
| SAM Automatic | 1,100 defect-present images | {auto_dice:.4f} | {sam_test['methods']['automatic']['summary']['mean_present_iou']:.4f} |

All methods use the same 1,100 defect-present Test E images for the
mean defect-present Dice and IoU comparison.

## 3. Additional Test Metrics

### U-Net

- Global pixel Dice: {unet_metrics['global_pixel_dice']:.4f}
- Global pixel IoU: {unet_metrics['global_pixel_iou']:.4f}
- Precision: {unet_metrics['precision']:.4f}
- Recall: {unet_metrics['recall']:.4f}
- Mean false-positive pixels on defect-free images: {unet_metrics['mean_empty_fp_pixels']:.2f}
- Defect-free image false-positive rate: {unet_metrics['empty_image_false_positive_rate']:.4f}

### SAM Automatic

- Global pixel Dice: {sam_test['methods']['automatic']['summary']['global_pixel_dice']:.4f}
- Global pixel IoU: {sam_test['methods']['automatic']['summary']['global_pixel_iou']:.4f}
- Precision: {sam_test['methods']['automatic']['summary']['global_precision']:.4f}
- Recall: {sam_test['methods']['automatic']['summary']['global_recall']:.4f}
- Mean false-positive pixels on defect-free images: {sam_test['methods']['automatic']['summary']['mean_empty_fp_pixels']:.2f}
- Defect-free image false-positive rate: {sam_test['methods']['automatic']['summary']['empty_image_false_positive_rate']:.4f}

The precision and recall aggregation scopes differ between methods.
Do not interpret these values as a fully matched comparison.

## 4. SAM Inference Time

| SAM method | Mean time per image |
|---|---:|
| Point oracle (embedding + prompt) | {sam_timing[0][1]:.4f} seconds |
| Box oracle (embedding + prompt) | {sam_timing[1][1]:.4f} seconds |
| Automatic mask generation | {sam_timing[2][1]:.4f} seconds |

These are measured SAM inference times. U-Net inference time was not
recorded in the supplied evaluation results, so no U-Net timing
comparison is reported.

## 5. Development vs Test E

| SAM method | Dev mean defect-present Dice | Test E mean defect-present Dice |
|---|---:|---:|
{chr(10).join(dev_test_lines)}

## 6. Main Findings

1. U-Net achieved the highest mean defect-present Dice on Test E
   ({unet_dice:.4f}), followed by SAM box prompting ({box_dice:.4f}).

2. SAM box prompting outperformed SAM point prompting. The box prompt
   provides more spatial information than a single positive point.

3. SAM automatic mask generation achieved a mean defect-present Dice
   of {auto_dice:.4f}. It has no semantic knowledge of which generated
   regions correspond to steel defects.

4. SAM automatic mask generation produced false positives on every
   defect-free Test E image. Its generated masks should therefore not
   be interpreted directly as defect predictions.

5. The U-Net achieved stronger defect segmentation, but it also
   produced false positives on {unet_metrics['empty_image_false_positive_rate'] * 100:.1f}%
   of defect-free images. This remains a significant limitation.

## 7. Important Methodological Limitations

### Oracle prompts

Point and box prompts are derived from the ground-truth masks. These
are oracle-prompted segmentation results, not independent human
annotation results. They quantify SAM's response when given accurate
spatial guidance.

### Automatic mask generation

SAM is class-agnostic. The experiment unions all generated masks and
compares this union with the binary defect mask. This is a proposal
coverage experiment, not a defect-aware classification pipeline.
The high false-positive rate reflects this limitation.

### Training data size

U-Net was trained using only 50 images. The findings therefore
describe this specific low-data experimental setting and should not
be generalized to all U-Net or SAM configurations.

### Dataset split

The human evaluation subset H is contained within Test E and is not
an independent test set. It is not included as a separate evaluation.

### Timing

SAM timing includes image embedding plus prompt inference for point
and box methods. Automatic timing covers automatic mask generation.
U-Net inference timing was not recorded in the supplied results.

## 8. Conclusion

In this experiment, a U-Net trained on 50 images achieved the strongest
defect-present segmentation performance on the held-out Test E set.

Among frozen SAM prompting strategies, box prompting performed better
than point prompting. Automatic mask generation had high recall but
very poor precision and substantial false positives, showing that
class-agnostic proposals alone are not sufficient for reliable steel
defect segmentation.

The results suggest that a small supervised model can be a stronger
direct defect segmenter than frozen, class-agnostic SAM in this
specific dataset and evaluation setup. However, oracle-prompted SAM
results should be interpreted as an upper-guidance comparison rather
than a deployable independent workflow.

## 9. Generated Figures

- dice_comparison.png
- iou_comparison.png
- empty_image_false_positives.png
- sam_inference_time.png
"""

report_path = REPORT_DIR / "final_research_results.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print("\n" + "=" * 80)
print("FINAL COMPARISON COMPLETED")
print("=" * 80)

print(f"CSV: {csv_path}")
print(f"Report: {report_path}")
print(f"Figures: {OUTPUT_DIR}")

print("\nBest mean defect-present Dice:")
print(f"{best_method['method']}: {best_method['dice']:.4f}")

print("\nAll comparison files generated successfully.")
