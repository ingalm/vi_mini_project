# This file is used to calculate the metrics of a loaded model
from ultralytics import YOLO

## History:¨
# Train17: YOLOv9t, no augmentation. Batch size 16
# Train19: YOLOv9s, no augmentation. Batch size 16
# Train20: YOLOv9s, with augmentation. Batch size 16
# Train27: YOLOv9s, with augmentation, transfer learning from weights. Batch size 32
# Train28: YOLOv9s, with augmentation and dropout, transfer learning. Batch size 32
# Train29: train28.pt, 150 more epochs.
# Train30: YOLOv9s, with custom old augmentation, transfer learning. Batch size 32
# Train31: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32
# Train32: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.8
# Train33: YOLOv9s, with custom new augmentation, transfer learning. Batch size 32, and cls weight 0.5, box weight 9, dfl weight 2

PATH = "runs/detect/train26/weights/best.pt"
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

# Evaluate the model on the test data
def test_real_time_inference(image_path):
    print(f"Running real-time inference on {image_path}")
    results = model.predict(image_path)
    results.show()  # Display results with bounding boxes

# test_real_time_inference(os.path.join(VALID_IMAGES_PATH, 'combined_image_96_png.rf.fc3591e525c1066cc77964dc56a47299.jpg'))
