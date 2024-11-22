import os
import shutil
from sklearn.model_selection import train_test_split
import albumentations as A
import cv2
from tqdm import tqdm
import numpy as np


PATH = "../../../projects/vc/data/ad/open/Poles"
SAVE_PATH = "./datasets"

addAugmentedSamples = True
cutIntoPatches = True  # Enable cutting into patches
PATCH_WIDTH = 128       # Define patch size (e.g., 128x128)

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
    Saves YOLO annotations back to file.
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

        # Save augmented image and labels
        augmented_image_name = f"aug_{image_name}"
        augmented_label_name = f"aug_{label_name}"
        cv2.imwrite(os.path.join(save_image_dir, augmented_image_name), augmented_image)
        save_annotations(os.path.join(save_label_dir, augmented_label_name), augmented_bboxes)


def cut_and_add_patches_to_dataset(images, labels, save_image_dir, save_label_dir, patch_width):
    """
    Cuts images into patches of specified height and width, and adds them to the dataset.
    Discards patches without bounding boxes. Optionally applies augmentations to patches if `addAugmentedSamples` is True.
    """
    image_height = 128
    
    for image_name, label_name in tqdm(zip(images, labels), total=len(images), desc="Cutting into patches"):
        image_path = os.path.join(PATH, "train/images", image_name)
        label_path = os.path.join(PATH, "train/labels", label_name)

        # Load image and annotations
        image = cv2.imread(image_path)
        bboxes = load_annotations(label_path)

        h, w, _ = image.shape

        for i in range(0, h, image_height):
            for j in range(0, w, patch_width):
                # Crop the patch
                patch = image[i:i + image_height, j:j + patch_width]
                if patch.shape[0] != image_height or patch.shape[1] != patch_width:
                    continue

                # Adjust bounding boxes for the patch
                patch_bboxes = []
                for bbox in bboxes:
                    x_center, y_center, width, height, label = bbox
                    abs_x_center, abs_y_center = x_center * w, y_center * h
                    abs_width, abs_height = width * w, height * h

                    if (
                        abs_x_center - abs_width / 2 >= j and
                        abs_x_center + abs_width / 2 <= j + patch_width and
                        abs_y_center - abs_height / 2 >= i and
                        abs_y_center + abs_height / 2 <= i + image_height
                    ):
                        # Normalize coordinates for the patch
                        new_x_center = (abs_x_center - j) / patch_width
                        new_y_center = (abs_y_center - i) / image_height
                        new_width = abs_width / patch_width
                        new_height = abs_height / image_height
                        patch_bboxes.append([new_x_center, new_y_center, new_width, new_height, label])

                # Save the patch only if it contains bounding boxes
                if patch_bboxes and np.random.rand() < 0.30:
                    patch_name = f"patch_{i}_{j}_{image_name}"
                    patch_label_name = f"patch_{i}_{j}_{label_name}"
                    cv2.imwrite(os.path.join(save_image_dir, patch_name), patch)
                    save_annotations(os.path.join(save_label_dir, patch_label_name), patch_bboxes)

                    # Apply augmentations to the patch if addAugmentedSamples is True
                    if addAugmentedSamples:
                        augmented = augmentation_pipeline(image=patch, bboxes=patch_bboxes)
                        augmented_image = augmented["image"]
                        augmented_bboxes = augmented["bboxes"]

                        # Save augmented patch and labels
                        aug_patch_name = f"aug_patch_{i}_{j}_{image_name}"
                        aug_label_name = f"aug_patch_{i}_{j}_{label_name}"
                        cv2.imwrite(os.path.join(save_image_dir, aug_patch_name), augmented_image)
                        save_annotations(os.path.join(save_label_dir, aug_label_name), augmented_bboxes)




# Augmentation pipeline
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=50, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ],
    bbox_params=A.BboxParams(format='yolo'),
)

if os.path.exists(SAVE_PATH):
    print("Removing old dataset files...")
    shutil.rmtree(SAVE_PATH)

for folder in folders:
    os.makedirs(f"{SAVE_PATH}/{folder}/images", exist_ok=True)
    os.makedirs(f"{SAVE_PATH}/{folder}/labels", exist_ok=True)

    if folder != "valid":
        images, labels = process_for_missing_images_or_labels(folder)

    if folder == "train":
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            images, labels, test_size=0.30, random_state=42
        )

        # Copy train subset
        for image, label in zip(train_images, train_labels):
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/train/images/{image}")
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/train/labels/{label}")

        # Add patches
        if cutIntoPatches:
            cut_and_add_patches_to_dataset(
                train_images,
                train_labels,
                f"{SAVE_PATH}/train/images",
                f"{SAVE_PATH}/train/labels",
                PATCH_WIDTH
            )
            
        # # Add augmented samples
        # if addAugmentedSamples:
        #     augment_and_add_to_dataset(
        #         train_images,
        #         train_labels,
        #         augmentation_pipeline,
        #         f"{SAVE_PATH}/train/images",
        #         f"{SAVE_PATH}/train/labels",
        #     )

        # Copy validation subset
        os.makedirs(f"{SAVE_PATH}/valid/images", exist_ok=True)
        os.makedirs(f"{SAVE_PATH}/valid/labels", exist_ok=True)
        for image, label in zip(valid_images, valid_labels):
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/valid/images/{image}")
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/valid/labels/{label}")

    if folder == "test":
        for image in images:
            shutil.copy(f"{PATH}/{folder}/images/{image}", f"{SAVE_PATH}/test/images/{image}")
        for label in labels:
            shutil.copy(f"{PATH}/{folder}/labels/{label}", f"{SAVE_PATH}/test/labels/{label}")

    print(f"Processed {folder} dataset")

print("Preprocessing complete")
