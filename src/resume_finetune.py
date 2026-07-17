import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
LAST_CKPT = str(Path('runs/segment/checkpoints/lp_30ep/weights/last.pt'))
EPOCHS = 30
BATCH = 12
IMGSZ = 640
SEED = 42
CLASS_NAMES = ['road', 'water', 'forest']

def main():
    print('Loading checkpoint...', flush=True)
    t0 = time.time()
    model = YOLOE(LAST_CKPT)
    model.set_classes(CLASS_NAMES, model.get_text_pe(CLASS_NAMES))
    print(f'Checkpoint loaded in {time.time() - t0:.1f}s', flush=True)
    print(f'\nResuming training ({EPOCHS} total epochs, batch={BATCH}, img={IMGSZ})...', flush=True)
    t0 = time.time()
    results = model.train(
        resume=True, data='finetune/data.yaml', epochs=EPOCHS, batch=BATCH,
        imgsz=IMGSZ, seed=SEED, trainer=YOLOEPESegTrainer, device=0,
        workers=0, amp=True, freeze=['backbone', 'neck', 'detection'],
        project='checkpoints', name='lp_30ep', exist_ok=True,
    )
    elapsed = time.time() - t0
    print(f'\nTraining completed in {elapsed:.0f}s ({elapsed / 60:.1f} min)', flush=True)
    final_path = Path('checkpoints/linear_probe_30ep.pt')
    model.save(str(final_path))
    print(f'Model saved to {final_path}', flush=True)

if __name__ == '__main__':
    main()
