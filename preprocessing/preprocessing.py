import os
import shutil
from sklearn.model_selection import train_test_split

# File intended to be used to preprocess the dataset for the YOLO models.
# The dataset is expected to be in the following structure:
# FOLDER STRUCTURE
# dataset
# ├── train
# ├──── images
# ├──── labels
# ├── test
# ├──── images
# ├──── labels
# ├── valid
# ├──── images
# ├──── labels
#
# The script will check for missing images or labels in the dataset and remove them from the dataset.
# Only one dataset will be saved in the datasets folder at a time.
# More preprocessing steps can be added as needed.

PATH = "original_dataset_YOLOv8" # Path to the dataset to be processed. To be changed if a new dataset is used.

folders = ['train', 'test', 'valid']


def process_for_missing_images_or_labels(folder):
    final_images = []
    final_labels = []
    
    images = os.listdir(f'{PATH}/{folder}/images')
    labels = os.listdir(f'{PATH}/{folder}/labels')
    
    for image in images:
        if image.replace(".jpg", ".txt") in labels:
            final_images.append(image)
            final_labels.append(image.replace(".jpg", ".txt"))
    
    return final_images, final_labels


# Save the processed dataset
SAVE_PATH = "datasets"

if os.path.exists(SAVE_PATH):
    print("Removing old dataset files...")
    shutil.rmtree(SAVE_PATH)


for folder in folders:
    os.makedirs(f"{SAVE_PATH}/{folder}/images", exist_ok=True)
    os.makedirs(f"{SAVE_PATH}/{folder}/labels", exist_ok=True)
    
    # Save images and labels, in new files
    images, labels = process_for_missing_images_or_labels(folder)
    
    #Split train data into train and valid subsets
    if folder == "train":
        # Split train data into train and validation subsets
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        # Copy train subset
        for image, label in zip(train_images, train_labels):
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/train/images/{image}")
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/train/labels/{label}")

        # Copy validation subset
        for image, label in zip(valid_images, valid_labels):
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/valid/images/{image}")
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/valid/labels/{label}")

        
    
    # Use valid data as test data
    if folder == "valid":
        for image in images:
            src = f"{PATH}/test/images/{image}"
            dest = f"{SAVE_PATH}/test/images/{image}"
            shutil.copy(src, dest)
        
        for label in labels:
            src = f"{PATH}/test/labels/{label}"
            dest = f"{SAVE_PATH}/test/labels/{label}"
            shutil.copy(src, dest)
    
    print(f"Processed {folder} dataset")
    
print("Preprocessing complete")
    