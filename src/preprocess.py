import csv, json, random, time
from pathlib import Path
import cv2
import numpy as np
CROP_SIZE = 640
STRIDE = 640
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = (0.75, 0.15, 0.10)
BASE = Path('C:\\Users\\X\\Desktop\\X\\seg')
HARMONIZED_MASKS = BASE / 'harmonized' / 'masks'
MANIFEST_SRC = BASE / 'harmonized' / 'manifest.csv'
CROPS_DIR = BASE / 'crops'
CROP_IMAGES = CROPS_DIR / 'images'
CROP_MASKS = CROPS_DIR / 'masks'
random.seed(SEED)

def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f'Cannot read mask: {path}')
    if mask.ndim != 2:
        mask = mask[:, :, 0] if mask.ndim == 3 else mask
    return mask

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def crop_grid(h, w):
    crops = []
    for y in range(0, h - CROP_SIZE + 1, STRIDE):
        for x in range(0, w - CROP_SIZE + 1, STRIDE):
            crops.append((x, y, CROP_SIZE, CROP_SIZE))
    return crops

def main():
    CROP_IMAGES.mkdir(parents=True, exist_ok=True)
    CROP_MASKS.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(MANIFEST_SRC) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f'Source manifest: {len(rows)} entries')
    all_crops = []
    crop_id = 0
    skipped = 0
    t0 = time.time()
    for i, entry in enumerate(rows):
        dataset = entry['dataset']
        split = entry['split']
        mask_name = entry['mask']
        source_img_path = entry['source_img']
        mask_path = HARMONIZED_MASKS / mask_name
        if not mask_path.exists():
            print(f'  Missing mask: {mask_path}')
            continue
        img_path = Path(source_img_path)
        if not img_path.exists():
            print(f'  Missing image: {img_path}')
            continue
        mask = load_mask(mask_path)
        img = load_image(img_path)
        h, w = mask.shape[:2]
        if img.shape[:2] != (h, w):
            print(f'  Size mismatch: {mask_name} mask={h}x{w} img={img.shape}')
            continue
        tiles = crop_grid(h, w)
        for x, y, tw, th in tiles:
            crop_id_str = f'crop_{crop_id:06d}'
            crop_img_name = f'{crop_id_str}.jpg'
            crop_mask_name = f'{crop_id_str}.png'
            img_crop_path = CROP_IMAGES / crop_img_name
            mask_crop_path = CROP_MASKS / crop_mask_name
            if img_crop_path.exists() and mask_crop_path.exists():
                skipped += 1
            else:
                img_crop = img[y:y + th, x:x + tw]
                mask_crop = mask[y:y + th, x:x + tw]
                cv2.imwrite(str(img_crop_path), img_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(mask_crop_path), mask_crop)
            all_crops.append({'crop_id': crop_id_str, 'source_dataset': dataset, 'source_split': split, 'source_image': mask_name.replace('.png', ''), 'crop_offset_x': x, 'crop_offset_y': y, 'crop_w': tw, 'crop_h': th, 'img_path': crop_img_name, 'mask_path': crop_mask_name})
            crop_id += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f'  {i + 1}/{len(rows)} sources -> {crop_id} crops ({skipped} skipped) in {elapsed:.0f}s', flush=True)
    print(f'\nTotal crops: {crop_id} ({skipped} skipped) in {time.time() - t0:.0f}s')
    test_crops = [c for c in all_crops if c['source_split'] == 'test']
    trainval_crops = [c for c in all_crops if c['source_split'] != 'test']
    random.shuffle(trainval_crops)
    n_tv = len(trainval_crops)
    n_train = int(n_tv * TRAIN_FRAC)
    n_val = int(n_tv * VAL_FRAC)
    for i, c in enumerate(trainval_crops):
        if i < n_train:
            c['split'] = 'train'
        elif i < n_train + n_val:
            c['split'] = 'val'
        else:
            c['split'] = 'test'
    for c in test_crops:
        c['split'] = 'test'
    all_crops = trainval_crops + test_crops
    manifest_path = CROPS_DIR / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        fields = ['crop_id', 'source_dataset', 'source_split', 'source_image', 'crop_offset_x', 'crop_offset_y', 'crop_w', 'crop_h', 'split', 'img_path', 'mask_path']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_crops)
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    for c in all_crops:
        split_counts[c['split']] += 1
    split_json = {'total_crops': len(all_crops), 'train': split_counts['train'], 'val': split_counts['val'], 'test': split_counts['test'], 'seed': SEED, 'crop_size': CROP_SIZE, 'stride': STRIDE, 'zero_fg_policy': 'keep_all'}
    split_path = CROPS_DIR / 'split.json'
    with open(split_path, 'w') as f:
        json.dump(split_json, f, indent=2)
    print(f'Manifest: {manifest_path}')
    print(f'Split: {split_json}')
    print(f'Split file: {split_path}')
if __name__ == '__main__':
    main()