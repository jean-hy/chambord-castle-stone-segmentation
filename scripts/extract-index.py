import os
import re
import shutil
from pathlib import Path

def extract_index(filename: str, prefix: str) -> str:
    """
    Extracts the numeric index from a filename given a specific prefix.
    For example, for filename 'patch_img_23.png' with prefix 'patch_img', it returns '23'.
    """
    pattern = re.compile(f"^{prefix}_(\\d+)\\.png$")
    match = pattern.match(filename)
    if match:
        return match.group(1)
    return None

def main():
    # Define directories relative to this script's location.
    script_dir = Path(__file__).resolve().parent
    images_dir = script_dir.parent / "data" / "patches-images-filter" / "final-images-without-filter"
    masks_dir = script_dir.parent / "data" / "patches-images-filter" / "final-patches-filter"

    # Define output directories for common pairs using the new folder name.
    output_dir = script_dir.parent / "data" / "patches-images-filter" / "final-patch-with-corresponding-image-filter"
    images_out = output_dir / "final-images"
    masks_out = output_dir / "final-patches"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Build dictionaries mapping index -> file path for images and masks.
    images_dict = {}
    for file in images_dir.glob("patch_img_*.png"):
        idx = extract_index(file.name, "patch_img")
        if idx is not None:
            images_dict[idx] = file

    masks_dict = {}
    for file in masks_dir.glob("patch_mask_*.png"):
        idx = extract_index(file.name, "patch_mask")
        if idx is not None:
            masks_dict[idx] = file

    # Identify common indices that exist in both dictionaries.
    common_indices = set(images_dict.keys()) & set(masks_dict.keys())
    print(f"Found {len(common_indices)} common pairs.")

    # Process common pairs: copy the matching images and masks into the output directories.
    for idx in common_indices:
        src_img = images_dict[idx]
        src_mask = masks_dict[idx]
        shutil.copy(src_img, images_out / src_img.name)
        shutil.copy(src_mask, masks_out / src_mask.name)

    print(f"Common pairs have been copied to: {output_dir}")

if __name__ == "__main__":
    main()
