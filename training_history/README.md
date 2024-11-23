# History


- Train17: YOLOv9t, no augmentation. Batch size 16
- Train19: YOLOv9s, no augmentation. Batch size 16
- Train20: YOLOv9s, with augmentation. Batch size 16
- Train27: YOLOv9s, with augmentation, transfer learning from weights. Batch size 32
- Train28: YOLOv9s, with augmentation and dropout, transfer learning. Batch size 32
- Train29: train28.pt, 150 more epochs.
- Train30: YOLOv9s, with custom old augmentation, transfer learning. Batch size 32
- Train31: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32
- Train32: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.8
- Train33_A: YOLOv9s, with costum new augmentation, transfer learning.
    'epochs': 150,
    'batch': 32,
    'device': str(device),
    'augment': True,
    "workers": 10,
    "dropout": 0.2,
    "cls": 0.5, 
    "dfl": 2, 
    "box": 9,
- Train34_A: yolo11 mAP@0.5:0.95: 0.271
- Train35_A: augmentation = false
- Train36_A: 200 epochs
- Train37_A: batch = 64
