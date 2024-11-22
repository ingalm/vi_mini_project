
import os
import shutil
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2
from tqdm import tqdm


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

PATH = "../../../projects/vc/data/ad/open/Poles"
# Save the processed dataset
SAVE_PATH = "datasets"

addAugmentedSamples = True

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

def load_annotations(label_path):
    """
    Loads YOLO formatted annotations and converts to Albumentations format.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    for line in lines:
        label, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height, int(label)])
    return bboxes

def save_annotations(label_path, bboxes):
    """
    Saves augmented YOLO annotations back to file.
    """
    with open(label_path, 'w') as file:
        for bbox in bboxes:
            label, x_center, y_center, width, height = bbox[-1], *bbox[:-1]
            file.write(f"{label} {x_center} {y_center} {width} {height}\n")

def augment_and_add_to_dataset(images, labels, aug_pipeline, save_image_dir, save_label_dir):
    """
    Applies augmentations and adds them as additional samples in the dataset.
    """
    for image_name, label_name in tqdm(zip(images, labels), total=len(images), desc="Augmenting"):
        image_path = os.path.join(PATH, "train/images", image_name)
        label_path = os.path.join(PATH, "train/labels", label_name)

        # Load image and annotations
        image = cv2.imread(image_path)
        bboxes = load_annotations(label_path)

        # Apply augmentations
        augmented = aug_pipeline(image=image, bboxes=bboxes)
        augmented_image = augmented["image"]
        augmented_bboxes = augmented["bboxes"]

        if augmented_bboxes: # Only save augmented samples with bounding boxes
            # Save augmented image and labels
            augmented_image_name = f"aug_{image_name}"
            augmented_label_name = f"aug_{label_name}"
            cv2.imwrite(os.path.join(save_image_dir, augmented_image_name), augmented_image)
            save_annotations(os.path.join(save_label_dir, augmented_label_name), augmented_bboxes)

# Old
# Augmentation pipeline
# augmentation_pipeline = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Blur(p=0.1),
#         A.Rotate(limit=15, p=0.5),
#     ],
#     bbox_params=A.BboxParams(format='yolo')
# )

# New
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomFog(p=0.1),
        A.RandomSnow(p=0.1),
        A.RandomRain(p=0.1),
        A.RandomShadow(p=0.1),
    ],
    bbox_params=A.BboxParams(format='yolo')
)

# Newnew
# augmentation_pipeline = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.Rotate(limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#     ],
#     bbox_params=A.BboxParams(format='yolo'),
# )

if os.path.exists(SAVE_PATH):
    print("Removing old dataset files...")
    shutil.rmtree(SAVE_PATH)

for folder in folders:
    os.makedirs(f"{SAVE_PATH}/{folder}/images", exist_ok=True)
    os.makedirs(f"{SAVE_PATH}/{folder}/labels", exist_ok=True)
    
    # Save images and labels, in new files
    if folder != "valid":
        images, labels = process_for_missing_images_or_labels(folder)
    
    # Split the dataset into train and validation subsets
    if folder == "train":
        # Split train data into train and validation subsets
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            images, labels, test_size=0.25, random_state=42
        )

        # Copy train subset
        for image, label in zip(train_images, train_labels):
            os.makedirs(f"{SAVE_PATH}/train/images", exist_ok=True)
            os.makedirs(f"{SAVE_PATH}/train/labels", exist_ok=True)
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/train/images/{image}")
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/train/labels/{label}")
            
        # Add augmented samples to training dataset
        if(addAugmentedSamples):
            augment_and_add_to_dataset(
                train_images,
                train_labels,
                augmentation_pipeline,
                f"{SAVE_PATH}/train/images",
                f"{SAVE_PATH}/train/labels",
            )

        # Copy validation subset
        os.makedirs(f"{SAVE_PATH}/valid/images", exist_ok=True)
        os.makedirs(f"{SAVE_PATH}/valid/labels", exist_ok=True)
        for image, label in zip(valid_images, valid_labels):
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/valid/images/{image}")
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/valid/labels/{label}")


    # Use test data as test data
    if folder == "test":
        os.makedirs(f"{SAVE_PATH}/test/images", exist_ok=True)
        os.makedirs(f"{SAVE_PATH}/test/labels", exist_ok=True)
        for image in images:
            src = f"{PATH}/{folder}/images/{image}"
            dest = f"{SAVE_PATH}/{folder}/images/{image}"
            shutil.copy(src, dest)
        
        for label in labels:
            src = f"{PATH}/{folder}/labels/{label}"
            dest = f"{SAVE_PATH}/{folder}/labels/{label}"
            shutil.copy(src, dest)
    
    print(f"Processed {folder} dataset")
    
print("Preprocessing complete")
    