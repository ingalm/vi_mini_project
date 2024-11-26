import os
from ultralytics import YOLO
import torch

### This program is intended to be used to train a YOLO model on a snow pole dataset.
### The dataset is expected to be preprocessed by the preprocessing_IDUN_data.py script.

# Automatically select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# YOLO model
model_name = 'yolo11s.pt'
model = YOLO(model_name)

# Training parameters
train_params = {
    'epochs': 150,
    'batch': 32,
    'device': str(device),
    'augment': True,
    "dropout": 0.2,
    "cls": 0.8,
    "optimizer": "sgd",
    'patience': 10,
    'imgsz': 1024,
    'mosaic': 1.0,
}

# Train the model
print("Training YOLO model...")
results = model.train(
    name=f"{model_name}_{train_params['optimizer']}_cut",
    data= 'data.yaml',
    epochs=train_params['epochs'],
    batch=train_params['batch'],
    device=train_params['device'],
    patience=20,
    augment=train_params['augment'],
    dropout=train_params['dropout'],
    cls=train_params['cls'],
    optimizer=train_params['optimizer'],
    imgsz=1024,
)
