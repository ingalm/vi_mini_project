# This file is used to calculate the metrics of a loaded model
from ultralytics import YOLO
from tabulate import tabulate
import numpy as np

models_to_evaluate = ["train17", "train19", "train20", "train27", "train29", "train30", "train31", "train32"]
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
