import json
import zlib
import base64
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.config import AERO_UNIFIED, AERO_BG_DEFAULT, UAVID_UNIFIED, UAVID_BG_DEFAULT, FLOODNET_UNIFIED, VDD_UNIFIED, SEMDRONE_UNIFIED, SEMDRONE_BG_DEFAULT, BG

def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    return img

def save_mask(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))

def mask_overlap_fraction(mask: np.ndarray) -> float:
    pass

def decode_bitmap(bitmap_data_b64: str) -> np.ndarray:
    z = zlib.decompress(base64.b64decode(bitmap_data_b64))
    arr = np.frombuffer(z, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    return img[:, :, 3].astype(bool)

def convert_json_bitmap(json_path: Path, img_h: int, img_w: int, class_title_to_unified: dict, bg_default: int=BG) -> np.ndarray:
    with open(json_path) as f:
        ann = json.load(f)
    canvas = np.full((img_h, img_w), bg_default, dtype=np.uint8)
    overlap_count = 0
    for obj in ann.get('objects', []):
        if obj.get('geometryType') != 'bitmap':
            continue
        title = obj.get('classTitle', '')
        unified_id = class_title_to_unified.get(title)
        if unified_id is None:
            continue
        mask = decode_bitmap(obj['bitmap']['data'])
        ox, oy = obj['bitmap']['origin']
        h, w = mask.shape
        region = canvas[oy:oy + h, ox:ox + w]
        overlap = (region != bg_default) & mask
        overlap_count += int(overlap.sum())
        region[mask] = unified_id
        canvas[oy:oy + h, ox:ox + w] = region
    return (canvas, overlap_count)

def convert_floodnet_mask(mask_path: Path) -> np.ndarray:
    mask = load_image(mask_path)
    assert mask.ndim == 2, f'FloodNet mask should be single-channel: {mask_path}'
    out = np.full_like(mask, BG, dtype=np.uint8)
    for src_id, dst_id in FLOODNET_UNIFIED.items():
        out[mask == src_id] = dst_id
    return out

def convert_vdd_mask(mask_path: Path) -> np.ndarray:
    mask = load_image(mask_path)
    assert mask.ndim == 2, f'VDD mask should be single-channel: {mask_path}'
    assert mask.max() <= 6, f'VDD mask has unexpected values: {np.unique(mask)}'
    out = np.full_like(mask, BG, dtype=np.uint8)
    for src_id, dst_id in VDD_UNIFIED.items():
        out[mask == src_id] = dst_id
    return out

def _load_semdrone_classdict(csv_path: Path):
    df = pd.read_csv(csv_path)
    rgb_to_name = {}
    for _, row in df.iterrows():
        name = row['name']
        rgb_to_name[int(row['red']), int(row['green']), int(row['blue'])] = name
    return rgb_to_name

def _build_semdrone_lut(rgb_to_name, name_to_unified, bg_default=BG):
    lut = np.full(256 * 256 * 256, bg_default, dtype=np.uint8)
    for (r, g, b), name in rgb_to_name.items():
        uid = name_to_unified.get(name)
        if uid is not None:
            idx = b << 16 | g << 8 | r
            lut[idx] = uid
    return lut
_SEMDRONE_LUT = None

def convert_semdrone_mask(mask_path: Path, rgb_to_name: dict, name_to_unified: dict, bg_default: int=BG) -> np.ndarray:
    global _SEMDRONE_LUT
    if _SEMDRONE_LUT is None:
        _SEMDRONE_LUT = _build_semdrone_lut(rgb_to_name, name_to_unified, bg_default)
    bgr_mask = load_image(mask_path)
    assert bgr_mask.ndim == 3 and bgr_mask.shape[2] >= 3, f'SemDrone mask should be RGB: {mask_path}'
    bgr_data = bgr_mask[:, :, :3].astype(np.uint32)
    encoded = bgr_data[:, :, 0] << 16 | bgr_data[:, :, 1] << 8 | bgr_data[:, :, 2]
    out = _SEMDRONE_LUT[encoded]
    return out

def get_image_dims(img_path: Path):
    img = load_image(img_path)
    return img.shape[:2]

def convert_dataset_sample(dataset: str, img_path: Path, ann_path: Path, **kwargs):
    if dataset in ('aeroscapes', 'uavid'):
        h, w = get_image_dims(img_path)
        class_map = AERO_UNIFIED if dataset == 'aeroscapes' else UAVID_UNIFIED
        mask, overlap = convert_json_bitmap(ann_path, h, w, class_map)
        return (mask, {'overlap_px': overlap})
    elif dataset == 'floodnet':
        mask = convert_floodnet_mask(ann_path)
        return (mask, {})
    elif dataset == 'vdd':
        mask = convert_vdd_mask(ann_path)
        return (mask, {})
    elif dataset == 'semdrone':
        rgb_to_name = kwargs['rgb_to_name']
        mask = convert_semdrone_mask(ann_path, rgb_to_name, SEMDRONE_UNIFIED)
        return (mask, {})
    else:
        raise ValueError(f'Unknown dataset: {dataset}')