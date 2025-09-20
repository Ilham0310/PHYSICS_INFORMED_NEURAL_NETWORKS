# pinn_blood_flow_local/preprocess_coco.py
import os
import json
import shutil
from PIL import Image, ImageDraw
import numpy as np
# from pycocotools import mask as coco_mask_utils # Not needed if segmentations are polygons
from tqdm import tqdm

# --- Configuration ---
# Path to your downloaded raw COCO dataset root (contains train, valid, test folders)
RAW_COCO_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IVUS')

# Output directory for processed images and masks
PROCESSED_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_ivus_dataset')

# Name of the annotation JSON file expected within each split folder (train, valid, test)
COCO_ANNOTATION_FILENAME_IN_SPLIT = "_annotations.coco.json" # Common Roboflow name

# Category ID in COCO JSON that represents the "lumen"
# !!! YOU MUST VERIFY THIS FROM YOUR DATASET !!!
# Based on your JSON, "Media" with id 1 is a candidate. If it's id 0, change this.
LUMEN_CATEGORY_ID_IN_JSON = 1 # UPDATE THIS IF "Media" id:0 is the lumen, or if you find a "lumen" category

# Pixel value to use for the lumen in the generated binary masks
MASK_LUMEN_PIXEL_VALUE = 255 # White for lumen

def create_mask_from_coco_segmentation(segmentation_data, img_height, img_width):
    """
    Creates a binary mask from COCO polygon segmentation data.
    Returns a NumPy array (binary mask).
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not isinstance(segmentation_data, list):
        print(f"Warning: Expected list for polygon segmentation, got {type(segmentation_data)}. Skipping segment.")
        return mask # Return empty mask

    for polygon_points_flat in segmentation_data:
        if not polygon_points_flat or len(polygon_points_flat) < 6 : # Need at least 3 points (6 values)
            # print(f"Warning: Invalid or empty polygon {polygon_points_flat}. Skipping.")
            continue 
        
        polygon_pil = []
        for i in range(0, len(polygon_points_flat), 2):
            try:
                x = int(round(polygon_points_flat[i]))
                y = int(round(polygon_points_flat[i+1]))
                polygon_pil.append((x, y))
            except (ValueError, TypeError):
                print(f"Warning: Invalid coordinate in polygon {polygon_points_flat}. Skipping point.")
                continue
        
        if len(polygon_pil) < 3:
            # print(f"Warning: Not enough valid points for polygon {polygon_pil}. Skipping.")
            continue

        try:
            # Create a temporary PIL Image from the mask to draw on
            # This ensures we are working with a fresh drawing surface for each polygon part
            # if segmentation_data contains multiple polygon lists for one instance (e.g. donut shape)
            # However, usually for one instance, segmentation_data is a list of *one* polygon list.
            # If ann['segmentation'] is [[poly1], [poly2]], this loop handles each poly.
            temp_pil_mask = Image.fromarray(np.zeros((img_height, img_width), dtype=np.uint8))
            img_draw = ImageDraw.Draw(temp_pil_mask)
            img_draw.polygon(polygon_pil, outline=MASK_LUMEN_PIXEL_VALUE, fill=MASK_LUMEN_PIXEL_VALUE)
            mask = np.maximum(mask, np.array(temp_pil_mask)) # Combine with main mask
        except Exception as e:
            print(f"Error drawing polygon {polygon_pil}: {e}")
            
    return mask

def process_coco_split(split_name, raw_dataset_root, processed_dataset_root):
    """
    Processes a single split (e.g., 'train', 'valid') of the COCO dataset.
    """
    split_input_path = os.path.join(raw_dataset_root, split_name)
    annotation_json_path = os.path.join(split_input_path, COCO_ANNOTATION_FILENAME_IN_SPLIT)

    if not os.path.isdir(split_input_path):
        print(f"Split directory not found: {split_input_path}. Skipping this split.")
        return 0
    if not os.path.exists(annotation_json_path):
        print(f"Annotation JSON not found for split '{split_name}' at: {annotation_json_path}. Skipping this split.")
        return 0

    # Create output directories for this split
    split_output_images_dir = os.path.join(processed_dataset_root, split_name, "images")
    split_output_masks_dir = os.path.join(processed_dataset_root, split_name, "masks")
    os.makedirs(split_output_images_dir, exist_ok=True)
    os.makedirs(split_output_masks_dir, exist_ok=True)

    print(f"\nProcessing split: '{split_name}'")
    print(f"Loading COCO annotations from: {annotation_json_path}")
    with open(annotation_json_path, 'r') as f:
        coco_data = json.load(f)

    # Check for categories (even though we use ID directly, it's good for user to know)
    if 'categories' not in coco_data:
        print(f"Warning: 'categories' field not found in {annotation_json_path}.")
    else:
        found_lumen_category = False
        for category in coco_data['categories']:
            if category['id'] == LUMEN_CATEGORY_ID_IN_JSON:
                print(f"Confirmed: Category ID {LUMEN_CATEGORY_ID_IN_JSON} corresponds to name '{category['name']}'. This will be used for lumen masks.")
                found_lumen_category = True
                break
        if not found_lumen_category:
            print(f"WARNING: Category ID {LUMEN_CATEGORY_ID_IN_JSON} (set for lumen) was NOT found in the JSON categories.")
            print("Available categories in this JSON:", [(cat['id'], cat['name']) for cat in coco_data['categories']])
            print("Please verify LUMEN_CATEGORY_ID_IN_JSON setting.")


    images_info = {img['id']: img for img in coco_data.get('images', [])}
    if not images_info:
        print(f"No 'images' found in {annotation_json_path} for split '{split_name}'.")
        return 0

    num_masks_created_for_split = 0
    annotations_for_split = coco_data.get('annotations', [])
    
    # Group annotations by image_id to combine masks if multiple lumen instances exist for one image
    # (though for lumen, typically there's only one main instance per image)
    image_id_to_annotations = {}
    for ann in annotations_for_split:
        if ann['category_id'] == LUMEN_CATEGORY_ID_IN_JSON:
            img_id = ann['image_id']
            if img_id not in image_id_to_annotations:
                image_id_to_annotations[img_id] = []
            image_id_to_annotations[img_id].append(ann)

    print(f"Processing {len(image_id_to_annotations)} images with lumen annotations for split '{split_name}'...")
    for image_id, image_anns in tqdm(image_id_to_annotations.items(), desc=f"Processing images in {split_name}"):
        if image_id not in images_info:
            print(f"Warning: Annotations for image_id {image_id} found, but image info is missing. Skipping.")
            continue

        img_info = images_info[image_id]
        img_filename = img_info['file_name']
        img_height = img_info['height']
        img_width = img_info['width']
        
        original_image_path = os.path.join(split_input_path, img_filename) # Images are in the same folder as JSON
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image file not found: {original_image_path}. Skipping.")
            continue
        
        # Initialize a combined mask for this image
        combined_binary_mask_np = np.zeros((img_height, img_width), dtype=np.uint8)

        for ann_for_image in image_anns: # Iterate through all lumen annotations for this image
            segmentation_data = ann_for_image['segmentation']
            # This function now handles multiple polygon lists if segmentation_data is like [[poly1], [poly2]]
            single_instance_mask_np = create_mask_from_coco_segmentation(segmentation_data, img_height, img_width)
            if single_instance_mask_np is not None:
                combined_binary_mask_np = np.maximum(combined_binary_mask_np, single_instance_mask_np) # Combine masks using maximum

        if np.any(combined_binary_mask_np): # If any part of the mask was drawn
            # Save the original image to processed directory for this split
            dest_image_path = os.path.join(split_output_images_dir, img_filename)
            shutil.copy(original_image_path, dest_image_path)

            # Save the generated combined binary mask
            mask_output_filename = os.path.splitext(img_filename)[0] + ".png" # Save masks as PNG
            dest_mask_path = os.path.join(split_output_masks_dir, mask_output_filename)
            
            mask_pil_to_save = Image.fromarray(combined_binary_mask_np)
            mask_pil_to_save.save(dest_mask_path)
            num_masks_created_for_split +=1
            
    print(f"Split '{split_name}' processing complete. {num_masks_created_for_split} lumen masks generated.")
    return num_masks_created_for_split

def main_preprocess():
    print("Starting COCO dataset preprocessing for train/valid/test splits...")
    print("Please ensure you have updated the following constants at the top of this script:")
    print(f" - RAW_COCO_DATASET_ROOT (current: {RAW_COCO_DATASET_ROOT})")
    print(f" - PROCESSED_DATASET_ROOT (current: {PROCESSED_DATASET_ROOT})")
    print(f" - COCO_ANNOTATION_FILENAME_IN_SPLIT (current: {COCO_ANNOTATION_FILENAME_IN_SPLIT})")
    print(f" - LUMEN_CATEGORY_ID_IN_JSON (current: {LUMEN_CATEGORY_ID_IN_JSON}) - VERIFY THIS!")
    print("-" * 70)

    total_masks_generated = 0
    for split in ["train", "valid", "test"]: # Add/remove splits as needed
        if os.path.exists(os.path.join(RAW_COCO_DATASET_ROOT, split)):
            total_masks_generated += process_coco_split(split, RAW_COCO_DATASET_ROOT, PROCESSED_DATASET_ROOT)
        else:
            print(f"Split directory for '{split}' not found in {RAW_COCO_DATASET_ROOT}. Skipping.")
            
    print(f"\n\nOverall preprocessing complete. Total {total_masks_generated} lumen masks generated across all processed splits.")
    print(f"Processed data saved in subfolders within: {PROCESSED_DATASET_ROOT}")

if __name__ == "__main__":
    main_preprocess()
