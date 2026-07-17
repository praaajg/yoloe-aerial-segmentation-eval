# YOLOE Segmentation Evaluation — Comparison Report

## Models Compared
| Method | Epochs | Batch | Description |
|---|---|---|---|
| **Zero-shot** | 0 | — | Pretrained YOLOE-26s-seg with optimized prompts at conf=0.10 |
| **Linear Probe** | 10 | 12 | YOLOEPESegTrainer + freeze(backbone+neck+detection), conf=0.25 |
| **Linear Probe (30ep)** | 30 | 12 | Same config, resume from 10-epoch checkpoint, conf=0.25 |

## Data
- 5 drone aerial segmentation datasets (Aeroscapes, FloodNet, UAVid, VDD, Semantic Drone)
- Unified 4-class: background(0), road(1), water(2), forest(3)
- Source-image-level split: 75/15/10 (no train/val/test leakage)
- 640×640 grid crops: 102,938 total (66,199 train, 13,239 val, 23,500 test)
- Polygon labels: 79,438 instances

## Test Metrics (pixel-level mIoU, dense prediction)

| Metric | Zero-shot | Linear Probe (10ep) | Linear Probe (30ep) |
|---|---|---|---|
| Pixel Accuracy | 0.4128 | 0.8331 | **0.8488** |
| mIoU | 0.2188 | 0.6731 | **0.7008** |
| Background IoU | 0.3142 | 0.7706 | **0.7907** |
| Road IoU | 0.2170 | 0.6582 | **0.6814** |
| Water IoU | 0.1043 | 0.6308 | **0.6808** |
| Forest IoU | 0.2397 | 0.6329 | **0.6502** |

## Training Progression (val set, mask mAP50-95)
| Epoch | mAP50(M) | mAP50-95(M) |
|---|---|---|
| 1 | 0.433 | 0.300 |
| 5 | 0.578 | 0.419 |
| 10 | 0.632 | 0.465 |
| 15 | 0.650 | 0.482 |
| 21 | 0.658 | 0.489 |
| 30 | **0.669** | **0.499** |

## Key Observations
1. Linear probing delivers 3.2× mIoU improvement over zero-shot (0.701 vs 0.219) and 2.1× PA improvement (0.849 vs 0.413).
2. Extending from 10→30 epochs yields modest gains: test mIoU rises 0.673→0.701 (+4.1% relative).
3. Water benefits most from extended training (0.631→0.681, +7.9%).
4. Zero-shot is poorly calibrated for the aerial domain (mean conf ~0.07–0.19 at conf=0.25), requiring conf=0.10 with optimized prompts for reasonable detection.
5. Training only the classification head (~13% of parameters) is highly effective for adapting YOLOE to the drone aerial domain.
6. The zero-shot gap is most pronounced for water (mIoU 0.104), likely due to the diversity of water appearances in aerial views vs typical ground-level training data.
