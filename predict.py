import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLOE

IMG_PATH = 'path/to/your/image.jpg'
CHECKPOINT = 'checkpoints/linear_probe_30ep.pt'
CONF = 0.25
CLASS_NAMES = ['road', 'water', 'forest']
YOLO_TO_SEMANTIC = {0: 1, 1: 2, 2: 3}
CLASS_COLORS = {0: (0, 0, 0), 1: (128, 64, 128), 2: (255, 0, 0), 3: (0, 128, 0)}
CLASS_LABELS = {0: 'background', 1: 'road', 2: 'water', 3: 'forest'}

model = YOLOE(CHECKPOINT)
model.set_classes(CLASS_NAMES, model.get_text_pe(CLASS_NAMES))

img_bgr = cv2.imread(IMG_PATH)
if img_bgr is None:
    raise FileNotFoundError(f'Could not read image: {IMG_PATH}')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_bgr.shape[:2]

results = model.predict(img_rgb, conf=CONF, retina_masks=False, verbose=False, max_det=100)
result = results[0]
pred = np.zeros((h, w), dtype=np.uint8)
if result.masks is not None:
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    for idx in np.argsort(confs):
        cls_id = classes[idx]
        if cls_id not in YOLO_TO_SEMANTIC:
            continue
        mask = masks[idx].astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred[mask] = YOLO_TO_SEMANTIC[cls_id]

overlay = np.zeros((h, w, 3), dtype=np.uint8)
for sem_id, color in CLASS_COLORS.items():
    overlay[pred == sem_id] = color

blended = cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0)
cv2.imwrite('output.png', blended)
print('Saved output.png', flush=True)
