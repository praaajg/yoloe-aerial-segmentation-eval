import warnings
warnings.filterwarnings('ignore')
import csv, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/yoloe-26s-seg.pt'
RETINA_MASKS = (sys.argv[2].lower() == 'true') if len(sys.argv) > 2 else True
CLASS_NAMES = ['road street highway asphalt concrete', 'water river lake sea ocean', 'forest trees woodland jungle']
YOLO_TO_SEMANTIC = {0: 1, 1: 2, 2: 3}
CONF = 0.10
NUM_CLASSES = 4
BATCH = 4 if RETINA_MASKS else 16
MB = 100

with open('crops/manifest.csv') as f:
    test_rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
print(f'Test crops: {len(test_rows)}', flush=True)

img_dir = Path('crops/images')
mask_dir = Path('crops/masks')
img_paths = [str(img_dir / r['img_path']) for r in test_rows]
mask_paths = [str(mask_dir / r['mask_path']) for r in test_rows]

t0 = time.time()
model = YOLOE(CHECKPOINT)
model.set_classes(CLASS_NAMES, model.get_text_pe(CLASS_NAMES))
print(f'Model loaded: {time.time()-t0:.1f}s', flush=True)

per_class_info = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(NUM_CLASSES)}
total_pixels = 0
correct_pixels = 0
total = len(test_rows)

t0 = time.time()
for start_mb in range(0, total, MB):
    end_mb = min(start_mb + MB, total)
    mb_t0 = time.time()
    mb_imgs = [cv2.imread(p) for p in img_paths[start_mb:end_mb]]
    mb_masks = mask_paths[start_mb:end_mb]
    for result, mask_path in zip(model.predict(mb_imgs, conf=CONF, retina_masks=RETINA_MASKS, verbose=False, batch=BATCH, max_det=100, stream=True), mb_masks):
        gt = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        gt = gt.squeeze()
        h, w = result.orig_shape
        pred = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for pred_idx in np.argsort(confs):
                cls_id = classes[pred_idx]
                if cls_id not in YOLO_TO_SEMANTIC:
                    continue
                mask = masks[pred_idx].astype(bool)
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                pred[mask] = YOLO_TO_SEMANTIC[cls_id]
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        for cls_id in range(NUM_CLASSES):
            pred_cls = pred == cls_id
            gt_cls = gt == cls_id
            tp = np.logical_and(pred_cls, gt_cls).sum()
            fp = pred_cls.sum() - tp
            fn = gt_cls.sum() - tp
            per_class_info[cls_id]['tp'] += int(tp)
            per_class_info[cls_id]['fp'] += int(fp)
            per_class_info[cls_id]['fn'] += int(fn)
        total_pixels += int(pred.size)
        correct_pixels += int(np.sum(pred == gt))
    mb_t = time.time() - mb_t0
    rate = (end_mb - start_mb) / mb_t if mb_t > 0 else 0
    elapsed = time.time() - t0
    overall_rate = (end_mb) / elapsed if elapsed > 0 else 0
    eta = (total - end_mb) / overall_rate if overall_rate > 0 else 0
    print(f'  {end_mb}/{total}  {mb_t:.0f}s  {rate:.1f}img/s  eta:{eta:.0f}s', flush=True)

per_class_iou = {}
for cls_id in range(NUM_CLASSES):
    tp = per_class_info[cls_id]['tp']
    fp = per_class_info[cls_id]['fp']
    fn = per_class_info[cls_id]['fn']
    denom = tp + fp + fn
    per_class_iou[cls_id] = tp / denom if denom > 0 else float('nan')

iou_vals = [v for v in per_class_iou.values() if not np.isnan(v)]
mean_iou = np.mean(iou_vals) if iou_vals else float('nan')
pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else float('nan')

results_dict = {
    'per_class_iou': {str(k): float(v) for k, v in per_class_iou.items()},
    'mean_iou': float(mean_iou),
    'pixel_accuracy': float(pixel_acc),
    'per_class_tp_fn_fp': {str(k): v for k, v in per_class_info.items()},
    'config': {'checkpoint': CHECKPOINT, 'conf': CONF, 'retina_masks': RETINA_MASKS, 'test_crops': total},
}

out_dir = Path('results')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'test_eval_metrics.json', 'w') as f:
    json.dump(results_dict, f, indent=2, default=str)

elapsed = time.time() - t0
print(f'\nmIoU={mean_iou:.4f}  PA={pixel_acc:.4f}  Time={elapsed:.0f}s')
for cls_id in range(NUM_CLASSES):
    name = ['background', 'road', 'water', 'forest'][cls_id]
    print(f'  {name}: IoU={per_class_iou[cls_id]:.4f}')
print(f'Saved results/test_eval_metrics.json')
