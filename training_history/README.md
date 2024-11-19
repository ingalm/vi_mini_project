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
- Train33: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.5, box weight 9, dfl weight 2
