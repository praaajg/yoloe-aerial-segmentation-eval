import sys, csv, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from src.config import HARMONIZED_MASKS, DATASET_SPLITS, AERO_UNIFIED, UAVID_UNIFIED, SEMDRONE_UNIFIED, SEMDRONE_BG_DEFAULT, SEMDRONE
from src.converters import convert_json_bitmap, convert_floodnet_mask, convert_vdd_mask, convert_semdrone_mask, _load_semdrone_classdict, get_image_dims, load_image, save_mask
SEMDRONE_RGB_TO_NAME = _load_semdrone_classdict(SEMDRONE / 'classes.csv')

def process_split(dataset_name, split_name, split_dir, img_subdir, ann_subdir):
    img_dir = split_dir / img_subdir
    ann_dir = split_dir / ann_subdir
    if not img_dir.is_dir() or not ann_dir.is_dir():
        return []
    img_files = sorted(img_dir.iterdir())
    rows = []
    for img_path in img_files:
        stem = img_path.stem
        if dataset_name in ('aeroscapes', 'uavid'):
            ann_path = ann_dir / f'{img_path.name}.json'
        elif dataset_name == 'floodnet':
            ann_path = ann_dir / f'{stem}_lab.png'
        elif dataset_name in ('vdd', 'semdrone'):
            ann_path = ann_dir / f'{stem}.png'
        if not ann_path.exists():
            continue
        try:
            if dataset_name in ('aeroscapes', 'uavid'):
                h, w = get_image_dims(img_path)
                class_map = AERO_UNIFIED if dataset_name == 'aeroscapes' else UAVID_UNIFIED
                mask, _ = convert_json_bitmap(ann_path, h, w, class_map)
            elif dataset_name == 'floodnet':
                mask = convert_floodnet_mask(ann_path)
            elif dataset_name == 'vdd':
                mask = convert_vdd_mask(ann_path)
            elif dataset_name == 'semdrone':
                mask = convert_semdrone_mask(ann_path, SEMDRONE_RGB_TO_NAME, SEMDRONE_UNIFIED, SEMDRONE_BG_DEFAULT)
        except Exception as e:
            print(f'    Error {ann_path.name}: {e}', flush=True)
            continue
        mask_name = f'{dataset_name}_{split_name}_{stem}.png'
        save_mask(mask, HARMONIZED_MASKS / mask_name)
        rows.append({'dataset': dataset_name, 'split': split_name, 'mask': mask_name, 'source_img': str(img_path), 'img_h': mask.shape[0], 'img_w': mask.shape[1]})
    return rows

def main():
    HARMONIZED_MASKS.mkdir(parents=True, exist_ok=True)
    existing_masks = {p.name for p in HARMONIZED_MASKS.glob('*.png')}
    all_rows = []
    total_processed = 0
    for dataset_name, splits in DATASET_SPLITS.items():
        print(f'\n=== {dataset_name} ===', flush=True)
        t0 = time.time()
        for split_name, (split_dir, img_subdir, ann_subdir) in splits.items():
            prefix = f'{dataset_name}_{split_name}_'
            existing = sum((1 for m in existing_masks if m.startswith(prefix)))
            img_dir = split_dir / img_subdir
            if img_dir.is_dir():
                total_imgs = len(list(img_dir.iterdir()))
            else:
                total_imgs = 0
            if existing >= total_imgs and total_imgs > 0:
                print(f'  {split_name}: {existing}/{total_imgs} already done, skipping', flush=True)
                continue
            rows = process_split(dataset_name, split_name, split_dir, img_subdir, ann_subdir)
            all_rows.extend(rows)
            total_processed += len(rows)
            print(f'  {split_name}: {len(rows)} done', flush=True)
        elapsed = time.time() - t0
        print(f'  Elapsed: {elapsed:.1f}s', flush=True)
    manifest_path = Path(__file__).resolve().parent.parent / 'harmonized' / 'manifest.csv'
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'split', 'mask', 'source_img', 'img_h', 'img_w'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nTotal new masks: {total_processed}', flush=True)
    print(f'Manifest: {manifest_path}', flush=True)
if __name__ == '__main__':
    main()