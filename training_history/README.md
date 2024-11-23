# History


- Train17: YOLOv9t, no augmentation. Batch size 16 (Transfer learning?)
- Train19: YOLOv9s, no augmentation. Batch size 16 (Transfer learning?)
- Train20: YOLOv9s, with augmentation. Batch size 16 (Transfer learning?)
- Train27: YOLOv9s, with augmentation, transfer learning from weights. Batch size 32
- Train28: YOLOv9s, with augmentation and dropout, transfer learning. Batch size 32
- Train29: train28.pt, 150 more epochs.
- Train30: YOLOv9s, with custom old augmentation, transfer learning. Batch size 32
- Train31: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32
- Train32: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.8
- Train34_mixup: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.5, box weight 9, dfl weight 2, and mixup 0.1
- Train38: New dataset with cut images. Includes all original images also. Augmented on both.
- Train39: Only cut images in train set, augmented. Cls=0.8
- Train41: Cut and normal images, all augmented. Cls=0.5, box=12, dfl=3
- Train42: Only augmented normal dataset. Cls=0.5, box=12, dfl=3
- Train46: Added rotation and vertical flip to augmentation. Base was Train32
- Train47: With rotation av vertical flip to augmentation. Base Train32. SGD optimizer
- Train47: Dataset from train32. SGD optimizer.