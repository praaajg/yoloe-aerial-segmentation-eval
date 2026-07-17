import cv2
import numpy as np
from src.config import BG, ROAD, WATER, FOREST
UNIFIED_TO_YOLO = {ROAD: 0, WATER: 1, FOREST: 2}
CLASS_PARAMS = {ROAD: {'min_area': 16, 'morph_kernel': None}, WATER: {'min_area': 16, 'morph_kernel': None}, FOREST: {'min_area': 100, 'morph_kernel': 5}}

def mask_to_polygons(mask: np.ndarray) -> list:
    h, w = mask.shape
    lines = []
    for unified_id in (ROAD, WATER, FOREST):
        yolo_id = UNIFIED_TO_YOLO[unified_id]
        params = CLASS_PARAMS[unified_id]
        min_area = params['min_area']
        kernel_size = params['morph_kernel']
        binary = (mask == unified_id).astype(np.uint8)
        if binary.sum() == 0:
            continue
        if kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            comp_mask = (labels == comp_id).astype(np.uint8)
            contours, hierarchy = cv2.findContours(comp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2).astype(np.float32)
            pts[:, 0] /= w
            pts[:, 1] /= h
            pts = np.clip(pts, 0, 1)
            line = f'{yolo_id} ' + ' '.join((f'{x:.6f} {y:.6f}' for x, y in pts))
            lines.append(line)
    return lines