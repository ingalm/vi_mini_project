import os
from ultralytics import YOLO
import torch

#PATH = "/cluster/projects/vc/data/ad/open/Poles"
PATH = "data"  # Main data path
IMAGE_INPUT_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 100


# Automatically select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dataset contains RGB images in .jpg format with labeeled snow poles stored in YOLOv9 format, in a .txt file.
# Test data is saved in the test folder, and training in the train folder
# Models should be leightweights and capable of handling real-time inference for edge device deployment
# Looking into small and tiny YOLO models first due to their lower computational requirements.

# Using established YOLO repositories such as YOLOv9 for training and inference

# Augmentation: Snowy conditions vary in visibility; apply augmentation like brightness adjustments, rotations, and horizontal flips to improve robustness
# Input size: Keep realitvely small to reduce computational requirements to optimize for speed and compatibility with edge devices.

# Strategy: 
# Begin with a pretrained model on COCO dataset, and fine-tune on the snow pole dataset, using transfer learning to reduce training time and improve accuracy.
# Try different YOLO models and compare performance to find the best model for the task.

# Evaluation metrics:
# – Precision
# – Recall
# – map@50
# – mAP@0.5:0.95
# These metrics can be calculated using YOLO's evaluation tools

# YOLO model
model = YOLO('yolov8n.pt')  # Using a pretrained Tiny model from COCO


# TRAIN_IMAGES_PATH = os.path.join(PATH, "train/images")
# TRAIN_LABELS_PATH = os.path.join(PATH, "train/labels")
# VALID_IMAGES_PATH = os.path.join(PATH, "valid/images")
# VALID_LABELS_PATH = os.path.join(PATH, "valid/labels")

# # Create a list of all image and label files in the training directory
# train_images = [os.path.join(TRAIN_IMAGES_PATH, f) for f in os.listdir(TRAIN_IMAGES_PATH) if f.endswith('.jpg')]
# train_labels = [os.path.join(TRAIN_LABELS_PATH, f) for f in os.listdir(TRAIN_LABELS_PATH) if f.endswith('.txt')]

# # Train-validation split within the training dataset
# train_images, val_images, train_labels, val_labels = train_test_split(
#     train_images, train_labels, test_size=0.2, random_state=42
# )

# Data configuration for YOLOv8
# data_config = {
#     'path': PATH,
#     'train': os.path.join(PATH, "train_split"),
#     'val': os.path.join(PATH, "val_split"),
#     'test': os.path.join(PATH, "test"),  # Using original validation set as test data
#     'nc': 1,  # Number of classes
#     'names': ['pole']  # Class name
# }

# Training parameters
train_params = {
    'imgsz': IMAGE_INPUT_SIZE,
    'epochs': EPOCHS,
    'batch': BATCH_SIZE,
    'device': str(device),  # Change to 'cpu' if you are not using a GPU
    'augment': True  # Enable augmentations (handled internally by YOLOv8)
}

# Train the model
print("Training YOLO model...")
results = model.train(
    data='data.yaml',
    imgsz=train_params['imgsz'],
    epochs=train_params['epochs'],
    batch=train_params['batch'],
    device=train_params['device']
)

model.save("best_snow_pole_detector.pt")

print("Evaluating YOLO model on test data...")
metrics = model.val(
    data="data.yaml",
    imgsz=train_params['imgsz'],
    split='test'  # Explicitly evaluate on the test (validation) set
)

# Display evaluation metrics
print("Evaluation Results:")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"mAP@50: {metrics['map50']:.3f}")
print(f"mAP@0.5:0.95: {metrics['map']:.3f}")

# Evaluate the model on the test data
def test_real_time_inference(image_path):
    print(f"Running real-time inference on {image_path}")
    results = model.predict(image_path)
    results.show()  # Display results with bounding boxes

# test_real_time_inference(os.path.join(VALID_IMAGES_PATH, 'combined_image_96_png.rf.fc3591e525c1066cc77964dc56a47299.jpg'))
