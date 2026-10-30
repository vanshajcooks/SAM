# Final Experimental Results

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
| U-Net (50 images) | 1,100 defect-present images | 0.4559 | 0.3209 |
| SAM Point (Oracle) | 1,100 defect-present images | 0.3434 | 0.2419 |
| SAM Box (Oracle) | 1,100 defect-present images | 0.4220 | 0.3164 |
| SAM Automatic | 1,100 defect-present images | 0.1054 | 0.0608 |

All methods use the same 1,100 defect-present Test E images for the
mean defect-present Dice and IoU comparison.

## 3. Additional Test Metrics

### U-Net

- Global pixel Dice: 0.4918
- Global pixel IoU: 0.3261
- Precision: 0.4024
- Recall: 0.6322
- Mean false-positive pixels on defect-free images: 49837.13
- Defect-free image false-positive rate: 0.9900

### SAM Automatic

- Global pixel Dice: 0.1042
- Global pixel IoU: 0.0550
- Precision: 0.0552
- Recall: 0.9371
- Mean false-positive pixels on defect-free images: 396138.54
- Defect-free image false-positive rate: 1.0000

The precision and recall aggregation scopes differ between methods.
Do not interpret these values as a fully matched comparison.

## 4. SAM Inference Time

| SAM method | Mean time per image |
|---|---:|
| Point oracle (embedding + prompt) | 0.1982 seconds |
| Box oracle (embedding + prompt) | 0.1949 seconds |
| Automatic mask generation | 0.8459 seconds |

These are measured SAM inference times. U-Net inference time was not
recorded in the supplied evaluation results, so no U-Net timing
comparison is reported.

## 5. Development vs Test E

| SAM method | Dev mean defect-present Dice | Test E mean defect-present Dice |
|---|---:|---:|
| SAM Point (Oracle) | 0.3810 | 0.3434 |
| SAM Box (Oracle) | 0.4538 | 0.4220 |
| SAM Automatic | 0.1179 | 0.1054 |

## 6. Main Findings

1. U-Net achieved the highest mean defect-present Dice on Test E
   (0.4559), followed by SAM box prompting (0.4220).

2. SAM box prompting outperformed SAM point prompting. The box prompt
   provides more spatial information than a single positive point.

3. SAM automatic mask generation achieved a mean defect-present Dice
   of 0.1054. It has no semantic knowledge of which generated
   regions correspond to steel defects.

4. SAM automatic mask generation produced false positives on every
   defect-free Test E image. Its generated masks should therefore not
   be interpreted directly as defect predictions.

5. The U-Net achieved stronger defect segmentation, but it also
   produced false positives on 99.0%
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
