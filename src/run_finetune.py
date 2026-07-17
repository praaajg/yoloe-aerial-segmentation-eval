import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
CHECKPOINT = str(Path('checkpoints/yoloe-v8s-seg.pt'))
DATA_YAML = str(Path('finetune/data.yaml'))
EPOCHS = 10
PATIENCE = 10
BATCH = 12
IMGSZ = 640
SEED = 42
CLASS_NAMES = ['road', 'water', 'forest']

def main():
    print('Loading model...', flush=True)
    t0 = time.time()
    model = YOLOE(CHECKPOINT)
    model.set_classes(CLASS_NAMES, model.get_text_pe(CLASS_NAMES))
    print(f'Model loaded in {time.time() - t0:.1f}s', flush=True)
    print('\nParameter audit (before training):', flush=True)
    trainable = sum((1 for p in model.parameters() if p.requires_grad))
    total = sum((1 for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'  Trainable: {trainable}/{total} tensors, {trainable_params:,}/{total_params:,} params', flush=True)
    print(f'\nStarting fine-tuning ({EPOCHS} epochs, batch={BATCH}, img={IMGSZ})...', flush=True)
    t0 = time.time()
    results = model.train(data=DATA_YAML, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, imgsz=IMGSZ, seed=SEED, trainer=YOLOEPESegTrainer, device=0, workers=0, augment=True, amp=True, freeze=['backbone', 'neck', 'detection'], project='checkpoints', name='lp_10ep', exist_ok=True)
    elapsed = time.time() - t0
    print(f'\nFine-tuning completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)', flush=True)
    final_path = Path('checkpoints/linear_probe_10ep.pt')
    model.save(str(final_path))
    print(f'Model saved to {final_path}', flush=True)
    print('\nParameter audit (after training):', flush=True)
    trainable = sum((1 for p in model.parameters() if p.requires_grad))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'  Trainable: {trainable} tensors, {trainable_params:,} params', flush=True)
if __name__ == '__main__':
    main()