
"""
Data Augmentation Script for Image-Mask Pairs

This script increases dataset diversity using techniques such as rotation,
flipping, zooming, and brightness adjustments. It loads original 256x256 images
and corresponding masks, applies the augmentation pipeline, and saves the
augmented outputs in new directories.
"""

import os
import cv2
import albumentations as A

def main():
    # -------------------------------------------------------------------------
    # 1. DEFINE DIRECTORIES
    # -------------------------------------------------------------------------
    orig_img_dir = '../data/patches/images'
    orig_mask_dir = '../data/patches/masks'

    aug_img_dir = '../data/augmented/images'
    aug_mask_dir = '../data/augmented/masks'
    
    # Create output directories if they do not exist.
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_mask_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 2. SET UP AUGMENTATION PIPELINE
    # -------------------------------------------------------------------------
    # This pipeline applies:
    # - Horizontal and vertical flips.
    # - Rotation (up to 90 degrees).
    # - Brightness and contrast adjustments.
    # - Random scaling (zooming) with a subsequent resize to maintain 256x256.
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Resize(height=256, width=256),
    ], additional_targets={'mask': 'mask'})
    
    # -------------------------------------------------------------------------
    # 3. PREPARE FILE LISTS AND SETTINGS
    # -------------------------------------------------------------------------
    # Filter to include only common image file extensions.
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_names = sorted([f for f in os.listdir(orig_img_dir) if f.lower().endswith(valid_extensions)])
    mask_names = sorted([f for f in os.listdir(orig_mask_dir) if f.lower().endswith(valid_extensions)])
    
    # Set the target total dataset size (e.g., 4000 images).
    target_dataset_size = 4000
    num_originals = len(image_names)
    num_aug_per_image = target_dataset_size // num_originals if num_originals > 0 else 0
    
    # Initialize counter for unique output filenames.
    count = 0
    
    # -------------------------------------------------------------------------
    # 4. AUGMENTATION LOOP
    # -------------------------------------------------------------------------
    for img_name, mask_name in zip(image_names, mask_names):
        img_path = os.path.join(orig_img_dir, img_name)
        mask_path = os.path.join(orig_mask_dir, mask_name)
        
        # Load image and mask.
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Skip this pair if the image or mask fails to load.
        if image is None or mask is None:
            print(f"Warning: Failed to load {img_path} or {mask_path}. Skipping.")
            continue
        
        # Save the original image and mask as part of the augmented dataset.
        cv2.imwrite(os.path.join(aug_img_dir, f'aug_{count}.png'), image)
        cv2.imwrite(os.path.join(aug_mask_dir, f'aug_{count}.png'), mask)
        count += 1
        
        # Generate and save augmented samples.
        for i in range(num_aug_per_image):
            augmented = transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            cv2.imwrite(os.path.join(aug_img_dir, f'aug_{count}.png'), aug_image)
            cv2.imwrite(os.path.join(aug_mask_dir, f'aug_{count}.png'), aug_mask)
            count += 1
            
            # Break early if we reach the target dataset size.
            if count >= target_dataset_size:
                break
        
        if count >= target_dataset_size:
            break
    
    # -------------------------------------------------------------------------
    # 5. COMPLETE
    # -------------------------------------------------------------------------
    print(f"Data augmentation completed. Total samples: {count}")

if __name__ == '__main__':
    main()
