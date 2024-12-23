# This file is used to calculate the metrics of a loaded model
from ultralytics import YOLO
from tabulate import tabulate
import numpy as np

#models_to_evaluate = ["8n", "8s", "8s_cls-2-sgd", "train17", "train19", "train30", "train31", "train32", "train39_cut_cls0.8", "train40A", "train41_cutandnormal_cls_dfl_box_changed", "train47_cls0.8_sgd_yolo9", "train50_cls0.8_sgd_yolo11", "train54_cls1.5_sgd_yolo11"]
models_to_evaluate = ["yolo11s.pt_adam", "yolo11s.pt_sgd", "yolov9s.pt_adam", "yolov9s.pt_sgd", "yolov8s.pt_adam", "yolov8s.pt_sgd", "yolov8n.pt_sgd"]


# - Train17: YOLOv9t, no augmentation. Batch size 16 (Transfer learning?)
# - Train19: YOLOv9s, no augmentation. Batch size 16 (Transfer learning?)
# - Train20: YOLOv9s, with augmentation. Batch size 16 (Transfer learning?)
# - Train27: YOLOv9s, with augmentation, transfer learning from weights. Batch size 32
# - Train28: YOLOv9s, with augmentation and dropout, transfer learning. Batch size 32
# - Train29: train28.pt: YOLOv9s, with augmentation and dropout, transfer learning. Batch size 32., 150 more epochs.
# - Train30: YOLOv9s, with custom old augmentation, transfer learning. Batch size 32
# - Train31: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32
# - Train32: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.8
# - Train39: Only cut images in train set, augmented. Cls=0.8
# - Train41: Cut and normal images, all augmented. Cls=0.5, box=12, dfl=3
# - Train47: With rotation av vertical flip to augmentation. Base Train32. SGD optimizer
# - Train47: Dataset from train32. SGD optimizer.
# - Train50: Dataset from train32. SGD optimizer. YOLO11s.pt
# - Train54: Same as train50, but cls=1.5

metrics = []

for i in models_to_evaluate:
    path = f"./runs/detect/{i}/weights/best.pt"
    model = YOLO(path)
    
    model_metrics = model.val(
        data="data.yaml",
        split='test'
    )
    
    extracted_metrics = [model_metrics.box.p, model_metrics.box.r, model_metrics.box.map50, model_metrics.box.map]
    
    metrics.append(extracted_metrics)
    

table_data = []
headers = ["Model", "Precision", "Recall", "mAP@50", "mAP@50-95"]

max_values = [max(column) for column in zip(*metrics)]
print(max_values)

for i, model in enumerate(models_to_evaluate):
    row = [model]
    for j, value in enumerate(metrics[i]):
        if isinstance(value, np.ndarray):
            value = value.item()  # Extract scalar from single-element array
        if value == max_values[j]:
            row.append(f"\033[92m{value:.2f}\033[0m")  # Green text for max value
        else:
            row.append(f"{value:.2f}")
    table_data.append(row)

print(tabulate(table_data, headers=headers, tablefmt="pretty"))
