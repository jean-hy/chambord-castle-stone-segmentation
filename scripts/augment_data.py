# Script for Data Augmentation of Image-Mask Pairs.

# This script increases dataset diversity using techniques such as rotation,
# flipping, zooming, and brightness adjustments. It loads original 256x256 images
# and masks from designated folders, applies the augmentation pipeline, and
# saves the augmented outputs in new directories.


import os
import cv2
import albumentations as A

def main():
    # Define directories for original images/masks and augmented outputs
    orig_img_dir = '../data/patches/images'
    orig_mask_dir = '../data/patches/masks'
    aug_img_dir = '../data/augmented/images'
    aug_mask_dir = '../data/augmented/masks'

    # Create output directories if they do not exist
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)

    # Define the augmentation pipeline using Albumentations
    # The pipeline includes:
    # - Horizontal and vertical flips for mirroring the images.
    # - Rotation up to 90 degrees for orientation variance.
    # - Random brightness and contrast adjustments.
    # - Zooming: Randomly scales the image (zooming in/out) and then resizes back to 256x256.
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # Zooming transformation: Random scale with up to 20% variation.
        A.RandomScale(scale_limit=0.2, p=0.5),
        # Resize to maintain original 256x256 dimensions after scaling
        A.Resize(height=256, width=256),
    ], additional_targets={'mask': 'mask'})

    # Retrieve sorted filenames from the original directories to ensure correct pairing
    image_names = sorted(os.listdir(orig_img_dir))
    mask_names = sorted(os.listdir(orig_mask_dir))

    # Set the target total dataset size.
    # With around 1100 original pairs, we need to generate additional samples
    # to reach around 4000 images.
    target_dataset_size = 4000
    num_originals = len(image_names)
    
    # Calculate number of augmentations per image pair.
    # (Integer division ensures a whole number; adjust if needed.)
    num_aug_per_image = target_dataset_size // num_originals

    # Counter for naming augmented files uniquely
    count = 0

    # Process each original image and its corresponding mask
    for img_name, mask_name in zip(image_names, mask_names):
        # Construct full file paths for the image and mask
        img_path = os.path.join(orig_img_dir, img_name)
        mask_path = os.path.join(orig_mask_dir, mask_name)
        
        # Read the image and mask; assume mask is grayscale
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Save the original image and mask in the augmented dataset as well
        cv2.imwrite(os.path.join(aug_img_dir, f'aug_{count}.png'), image)
        cv2.imwrite(os.path.join(aug_mask_dir, f'aug_{count}.png'), mask)
        count += 1

        # Apply the augmentation pipeline to create additional samples
        for i in range(num_aug_per_image):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Save the augmented image and mask with unique filenames
            cv2.imwrite(os.path.join(aug_img_dir, f'aug_{count}.png'), aug_image)
            cv2.imwrite(os.path.join(aug_mask_dir, f'aug_{count}.png'), aug_mask)
            count += 1
            
            # Break early if the target dataset size is reached
            if count >= target_dataset_size:
                break
        if count >= target_dataset_size:
            break

    print(f"Data augmentation completed. Total samples: {count}")

if __name__ == '__main__':
    main()