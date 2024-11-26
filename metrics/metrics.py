# This file is used to calculate the metrics of a loaded model
from ultralytics import YOLO

PATH = "./runs/detect/yolov9s.pt_sgd/weights/best.pt"
model = YOLO(PATH)

print("Evaluating YOLO model on test data...")
metrics = model.val(
    data="data.yaml",
    split='test'  # Explicitly evaluate on the test (validation) set
)

# Display evaluation metrics
print("Evaluation Results:")
print(f"Precision (mean): {metrics.box.p.mean():.3f}")  # Mean precision across classes
print(f"Recall (mean): {metrics.box.r.mean():.3f}")     # Mean recall across classes
print(f"mAP@50: {metrics.box.map50:.3f}")              # mAP@50
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")          # mAP@0.5:0.95