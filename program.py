import os
from ultralytics import YOLO
import torch

### This program is intended to be used to train a YOLO model on a snow pole dataset.
### The dataset is expected to be preprocessed by the preprocessing.py script.

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
# Try different YOLO models and compare performance to find the best model for the task.
# Try to use transfer learning to speed up training and improve performance.


# YOLO model
model = YOLO('yolov9s.pt')  # Using a pretrained Tiny model from COCO


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
    'epochs': 150,
    'batch': 32,
    'device': str(device),
    'augment': False,
    "workers": 10,
    "dropout": 0.2,
    "optimizer": "adamW",
    "cls": 0.8, # Class weight. To account for class imbalance
    "dfl": 2, # Default class weight
    "box": 9, # Box loss weight
}

# Next plans for training:
# Try ground up training with a small model
# Try YOLOv11 model

# Train the model
print("Training YOLO model...")
results = model.train(
    data= 'data.yaml',
    epochs=train_params['epochs'],
    batch=train_params['batch'],
    device=train_params['device'],
    patience=10,
    augment=train_params['augment'],
    workers=train_params['workers'],
    dropout=train_params['dropout'],
    cls=train_params['cls'],
    optimizer=train_params['optimizer'],
)
