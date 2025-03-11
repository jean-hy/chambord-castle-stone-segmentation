"""
Script for Data Augmentation of Image-Mask Pairs.

This script increases dataset diversity using techniques such as rotation,
flipping, zooming, and brightness adjustments. It loads original 256x256 images
and masks from designated folders, applies the augmentation pipeline, and
saves the augmented outputs in new directories, ensuring an exact total 
of 'target_dataset_size' image-mask pairs.
"""

import os
import cv2
import albumentations as A

def main():
    # Directories for original images/masks
    orig_img_dir = '../data/patches/images'
    orig_mask_dir = '../data/patches/masks'
    # Output directories for augmented images/masks
    aug_img_dir = '../data/augmented/images'
    aug_mask_dir = '../data/augmented/masks'

    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    # Define the augmentation pipeline using Albumentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # Zooming transformation: random scale up to ±20%
        A.RandomScale(scale_limit=0.2, p=0.5),
        # Resize back to 256x256 after scaling
        A.Resize(height=256, width=256),
    ], additional_targets={'mask': 'mask'})

    # Retrieve sorted filenames to ensure correct pairing
    image_names = sorted(os.listdir(orig_img_dir))
    mask_names = sorted(os.listdir(orig_mask_dir))

    # Number of original image-mask pairs
    num_originals = len(image_names)

    # Desired total size of the augmented dataset
    target_dataset_size = 4000

    if num_originals == 0:
        print("No original images found. Exiting.")
        return

    # If target_dataset_size is less than number of originals, handle edge case:
    if target_dataset_size < num_originals:
        print(f"Target size {target_dataset_size} is smaller than the number of originals {num_originals}.")
        print("Exiting without augmentation.")
        return

    # Calculate how many additional samples we need (beyond the original images).
    additional_needed = target_dataset_size - num_originals

    # Evenly distribute these extra augmentations per image.
    num_aug_per_image_base = additional_needed // num_originals
    remainder = additional_needed % num_originals

    # We'll keep a counter to name files uniquely.
    count = 0

    # Process each original image and its corresponding mask
    for idx, (img_name, mask_name) in enumerate(zip(image_names, mask_names)):
        # Construct full file paths
        img_path = os.path.join(orig_img_dir, img_name)
        mask_path = os.path.join(orig_mask_dir, mask_name)

        # Read image (color) and mask (grayscale)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 1) Save the "original" as part of the dataset
        cv2.imwrite(os.path.join(aug_img_dir, f'aug_{count}.png'), image)
        cv2.imwrite(os.path.join(aug_mask_dir, f'aug_{count}.png'), mask)
        count += 1

        # 2) Compute how many augmentations for this particular image
        #    - base number for all images
        #    - +1 if idx < remainder
        aug_for_this_image = num_aug_per_image_base + (1 if idx < remainder else 0)

        # 3) Apply the augmentation pipeline 'aug_for_this_image' times
        for _ in range(aug_for_this_image):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']

            cv2.imwrite(os.path.join(aug_img_dir, f'aug_{count}.png'), aug_image)
            cv2.imwrite(os.path.join(aug_mask_dir, f'aug_{count}.png'), aug_mask)
            count += 1

    print(f"Data augmentation completed. Total samples: {count}")
    if count == target_dataset_size:
        print(f"Successfully created exactly {target_dataset_size} image-mask pairs.")
    else:
        print(f"Warning: final count ({count}) != target ({target_dataset_size}).")

if __name__ == '__main__':
    main()
