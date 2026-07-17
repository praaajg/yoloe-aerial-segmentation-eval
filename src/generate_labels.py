import sys, csv, time, random, statistics
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BASE, ROAD, WATER, FOREST
from src.polygons import mask_to_polygons

CROPS_DIR = BASE / 'crops'
FINETUNE_DIR = BASE / 'finetune'
LABEL_DIRS = {'train': FINETUNE_DIR / 'labels' / 'train', 'val': FINETUNE_DIR / 'labels' / 'val'}
IMAGE_DIRS = {'train': FINETUNE_DIR / 'images' / 'train', 'val': FINETUNE_DIR / 'images' / 'val'}
CLASS_NAMES = {ROAD: 'road', WATER: 'water', FOREST: 'forest'}

def sample_check(n=20):
    mask_dir = CROPS_DIR / 'masks'
    all_masks = sorted(mask_dir.glob('*.png'))
    random.seed(42)
    samples = random.sample(all_masks, min(n, len(all_masks)))
    print(f'\nPre-check: sampling {len(samples)} masks', flush=True)
    per_class = {ROAD: [], WATER: [], FOREST: []}
    for mpath in samples:
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        lines = mask_to_polygons(mask)
        for line in lines:
            parts = line.split()
            yolo_id = int(parts[0])
            unified = {0: ROAD, 1: WATER, 2: FOREST}[yolo_id]
            coords = list(map(float, parts[1:]))
            xs, ys = coords[0::2], coords[1::2]
            area = 0.0
            if len(xs) >= 3:
                img_pts = np.column_stack((xs, ys))
                area = cv2.contourArea(img_pts.astype(np.float32))
            per_class[unified].append(area)
    for cls_id, areas in per_class.items():
        n_inst = len(areas)
        if n_inst > 0:
            mean_a = statistics.mean(areas)
            median_a = statistics.median(areas)
        else:
            mean_a = median_a = 0
        print(f'  {CLASS_NAMES[cls_id]}: {n_inst} instances, mean_area={mean_a:.4f}, median_area={median_a:.4f}', flush=True)
    road_n = len(per_class[ROAD])
    water_n = len(per_class[WATER])
    forest_n = len(per_class[FOREST])
    max_n = max(road_n, water_n, forest_n)
    min_n = min(road_n, water_n, forest_n) if min(road_n, water_n, forest_n) > 0 else 1
    ratio = max_n / min_n
    if ratio > 10:
        print(f'WARNING: instance count ratio {ratio:.1f}x between classes -- possible fragmentation issue', flush=True)
    else:
        print(f'Instance count ratio: {ratio:.1f}x -- looks healthy', flush=True)
    print(flush=True)

def main():
    for d in list(LABEL_DIRS.values()) + list(IMAGE_DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)
    manifest_path = CROPS_DIR / 'manifest.csv'
    with open(manifest_path) as f:
        all_rows = list(csv.DictReader(f))
    work_rows = [r for r in all_rows if r['split'] in ('train', 'val')]
    n_train = sum((1 for r in work_rows if r['split'] == 'train'))
    n_val = sum((1 for r in work_rows if r['split'] == 'val'))
    print(f'Total crops to process: {len(work_rows)} ({n_train} train, {n_val} val)', flush=True)
    sample_check(20)
    t0 = time.time()
    done = 0
    skipped = 0
    per_class_total = {ROAD: {'count': 0, 'areas': []}, WATER: {'count': 0, 'areas': []}, FOREST: {'count': 0, 'areas': []}}
    for row in work_rows:
        split = row['split']
        mask_name = row['mask_path']
        img_name = row['img_path']
        crop_id = row['crop_id']
        label_path = LABEL_DIRS[split] / f'{crop_id}.txt'
        img_dst = IMAGE_DIRS[split] / img_name
        if label_path.exists() and img_dst.exists():
            skipped += 1
            done += 1
            continue
        mask = cv2.imread(str(CROPS_DIR / 'masks' / mask_name), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        lines = mask_to_polygons(mask)
        with open(label_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')
                parts = line.split()
                yolo_id = int(parts[0])
                unified = {0: ROAD, 1: WATER, 2: FOREST}[yolo_id]
                per_class_total[unified]['count'] += 1
                coords = list(map(float, parts[1:]))
                xs, ys = coords[0::2], coords[1::2]
                if len(xs) >= 3:
                    img_pts = np.column_stack((xs, ys))
                    area = cv2.contourArea(img_pts.astype(np.float32))
                    per_class_total[unified]['areas'].append(area)
        img_src = CROPS_DIR / 'images' / img_name
        if img_src.exists() and (not img_dst.exists()):
            img_data = cv2.imread(str(img_src))
            if img_data is not None:
                cv2.imwrite(str(img_dst), img_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1
        if done % 5000 == 0:
            print(f'  {done}/{len(work_rows)} in {time.time() - t0:.0f}s ({skipped} skipped)', flush=True)
    elapsed = time.time() - t0
    print(f'\nDone: {done} crops in {elapsed:.0f}s ({skipped} skipped)', flush=True)
    print(f'\nPer-class polygon summary (across all generated labels):', flush=True)
    total_polys = 0
    for cls_id, data in per_class_total.items():
        n = data['count']
        total_polys += n
        areas = data['areas']
        if n > 0:
            print(f'  {CLASS_NAMES[cls_id]}: {n} instances, mean_area={statistics.mean(areas):.4f}, median_area={statistics.median(areas):.4f}', flush=True)
        else:
            print(f'  {CLASS_NAMES[cls_id]}: 0 instances', flush=True)
    print(f'  Total polygons: {total_polys}', flush=True)
    yaml_path = FINETUNE_DIR / 'data.yaml'
    yaml_content = f'path: {FINETUNE_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest:\n\nnames:\n  0: road\n  1: water\n  2: forest\n'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f'data.yaml: {yaml_path}', flush=True)
    total_images = sum((len(list(d.glob('*.jpg'))) for d in IMAGE_DIRS.values()))
    total_labels = sum((len(list(d.glob('*.txt'))) for d in LABEL_DIRS.values()))
    print(f'Images: {total_images}, Labels: {total_labels}', flush=True)

if __name__ == '__main__':
    main()
