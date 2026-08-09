import sys
import torch
import ultralytics

sys.modules['ultralytics.yolo'] = ultralytics

ckpt = torch.load('models/license_plate_detector.pt', map_location='cpu', weights_only=False)

print("=== CHECKPOINT KEYS ===")
print(ckpt.keys())

if 'train_args' in ckpt:
    print("\n=== TRAINING ARGS ===")
    args = ckpt['train_args']
    if hasattr(args, 'items'):
        for k, v in args.items():
            if k in ['model', 'data', 'epochs', 'batch', 'imgsz', 'optimizer']:
                print(f"{k}: {v}")
    elif type(args) is dict:
        for k, v in args.items():
            if k in ['model', 'data', 'epochs', 'batch', 'imgsz', 'optimizer']:
                print(f"{k}: {v}")
    else:
        print(args)
            
if 'model' in ckpt:
    model = ckpt['model']
    print("\n=== MODEL INFO ===")
    print(type(model))
    if hasattr(model, 'names'):
        print("Classes:", model.names)
    elif hasattr(model, 'module') and hasattr(model.module, 'names'):
         print("Classes:", model.module.names)
