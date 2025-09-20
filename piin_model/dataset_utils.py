# pinn_blood_flow_local/pinn_model/dataset_utils.py
import os
import glob
from PIL import Image
import numpy as np
from skimage import measure, morphology
from scipy.ndimage import binary_fill_holes

DEFAULT_PROCESSED_DATA_ROOT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    'processed_ivus_dataset'
))
IMAGE_SUBDIR_NAME_IN_SPLIT = "images"
MASK_SUBDIR_NAME_IN_SPLIT = "masks"
DEFAULT_TRAIN_SPLIT_NAME = "train"
LUMEN_PIXEL_VALUE_IN_MASK = 255

def get_processed_image_mask_pairs(dataset_root_path=DEFAULT_PROCESSED_DATA_ROOT_PATH,
                                   split_name=DEFAULT_TRAIN_SPLIT_NAME):
    split_data_path = os.path.join(dataset_root_path, split_name)
    image_folder = os.path.join(split_data_path, IMAGE_SUBDIR_NAME_IN_SPLIT)
    mask_folder = os.path.join(split_data_path, MASK_SUBDIR_NAME_IN_SPLIT)

    if not os.path.isdir(split_data_path):
        print(f"Error: Split directory '{split_data_path}' not found in processed dataset.")
        return []
    if not os.path.isdir(image_folder):
        print(f"Error: Processed image folder '{image_folder}' not found for split '{split_name}'.")
        return []
    if not os.path.isdir(mask_folder):
        print(f"Error: Processed mask folder '{mask_folder}' not found for split '{split_name}'.")
        return []

    image_files = sorted(glob.glob(os.path.join(image_folder, "*.*")))
    data_samples = []
    for img_fpath in image_files:
        basename = os.path.basename(img_fpath)
        name_part, _ = os.path.splitext(basename)
        mask_filename = f"{name_part}.png"
        mask_fpath = os.path.join(mask_folder, mask_filename)
        if os.path.exists(mask_fpath):
            data_samples.append({"image_path": img_fpath, "mask_path": mask_fpath})
        else:
            print(f"Warning: No mask found for image {img_fpath} (expected at {mask_fpath})")
    print(f"Found {len(data_samples)} image-mask pairs in '{split_data_path}'.")
    return data_samples

def extract_polygon_from_mask_file(mask_path, target_size=(128, 128), lumen_pixel_value=LUMEN_PIXEL_VALUE_IN_MASK, simplify_tolerance=0.005):
    try:
        mask_pil = Image.open(mask_path).convert("L")
        original_aspect_ratio = mask_pil.width / mask_pil.height
        if mask_pil.width > mask_pil.height:
            new_width = target_size[0]; new_height = int(new_width / original_aspect_ratio)
            if new_height > target_size[1]: new_height = target_size[1]; new_width = int(new_height * original_aspect_ratio)
        else:
            new_height = target_size[1]; new_width = int(new_height * original_aspect_ratio)
            if new_width > target_size[0]: new_width = target_size[0]; new_height = int(new_width / original_aspect_ratio)
        mask_resized_pil = mask_pil.resize((new_width, new_height), Image.NEAREST)
        mask_array_resized = np.array(mask_resized_pil)
        current_height, current_width = mask_array_resized.shape[0], mask_array_resized.shape[1]

        binary_lumen_mask = (mask_array_resized == lumen_pixel_value)
        binary_lumen_mask = binary_fill_holes(binary_lumen_mask)
        binary_lumen_mask = morphology.remove_small_objects(binary_lumen_mask, min_size=10) # Reduced min_size for smaller valid lumens
        if not np.any(binary_lumen_mask):
            # print(f"Warning (extract_polygon): No lumen pixels found in {mask_path} after processing.")
            return None
        
        contours = measure.find_contours(binary_lumen_mask, 0.5)
        if not contours:
            # print(f"Warning (extract_polygon): No contours found in {mask_path}.")
            return None
        contour = sorted(contours, key=len, reverse=True)[0]

        diag = np.sqrt(current_width**2 + current_height**2)
        simplified_contour = measure.approximate_polygon(contour, tolerance=simplify_tolerance * diag)

        if len(simplified_contour) < 3:
            # print(f"Warning (extract_polygon): Simplified contour for {mask_path} has < 3 vertices.")
            return None

        # Check for sufficient unique points
        unique_simplified_points = np.unique(simplified_contour, axis=0)
        if len(unique_simplified_points) < 3:
            print(f"Warning (extract_polygon): Simplified contour for {mask_path} has < 3 unique vertices ({len(unique_simplified_points)}). Polygon may be degenerate. Skipping.")
            return None
        
        # Check if points are nearly collinear (very basic check by looking at ranges of original contour pixels)
        # Use a small pixel tolerance. simplified_contour is in (row, col) pixel coords.
        y_coords_temp = simplified_contour[:, 0] # row
        x_coords_temp = simplified_contour[:, 1] # col
        pixel_tolerance_collinear = 2.0 # Allow for 2 pixels variation
        if (np.max(x_coords_temp) - np.min(x_coords_temp) < pixel_tolerance_collinear) or \
           (np.max(y_coords_temp) - np.min(y_coords_temp) < pixel_tolerance_collinear):
            print(f"Warning (extract_polygon): Simplified contour for {mask_path} appears to be collinear or a point (x_range: {np.max(x_coords_temp) - np.min(x_coords_temp)}, y_range: {np.max(y_coords_temp) - np.min(y_coords_temp)}). Skipping.")
            return None
            
        normalized_vertices = simplified_contour[:, [1, 0]] / np.array([current_width, current_height], dtype=np.float32)
        if not np.allclose(normalized_vertices[0], normalized_vertices[-1]):
            normalized_vertices = np.vstack([normalized_vertices, normalized_vertices[0]])
        
        x_coords_norm = normalized_vertices[:, 0]; y_coords_norm = normalized_vertices[:, 1]
        area = 0.5 * np.sum(x_coords_norm * np.roll(y_coords_norm, -1) - np.roll(x_coords_norm, -1) * y_coords_norm)
        if area < 0: normalized_vertices = normalized_vertices[::-1]
        
        return normalized_vertices.astype(np.float32) # Ensure float32 output
    except Exception as e:
        print(f"Error processing mask file {mask_path}: {e}"); import traceback; traceback.print_exc(); return None

if __name__ == '__main__':
    print(f"Looking for processed dataset at root: {DEFAULT_PROCESSED_DATA_ROOT_PATH}")
    _test_split = "train"
    print(f"Testing loading from split: '{_test_split}'")
    _test_pairs = get_processed_image_mask_pairs(split_name=_test_split)
    if _test_pairs:
        print(f"Found {len(_test_pairs)} processed image-mask pairs in split '{_test_split}'.")
        _first_mask = _test_pairs[0]['mask_path']
        print(f"Attempting to extract polygon from first mask: {_first_mask}")
        _poly = extract_polygon_from_mask_file(_first_mask)
        if _poly is not None: print(f"Extracted polygon with {len(_poly)} vertices from {_first_mask}. Shape: {_poly.shape}, Dtype: {_poly.dtype}")
        else: print(f"Failed to extract polygon from {_first_mask}")
    else:
        print(f"No processed image-mask pairs found in split '{_test_split}'.")

