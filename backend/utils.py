# pinn_blood_flow_local/backend/utils.py
from PIL import Image
import numpy as np
import io
from skimage import filters, measure, morphology
from scipy.ndimage import binary_fill_holes

# This function is for segmenting a RAW IVUS image during prediction
def extract_lumen_polygon_from_raw_image(image_bytes, target_size=(256,256), simplify_tolerance=0.005):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        original_aspect_ratio = image.width / image.height
        if image.width > image.height:
            new_width = target_size[0]; new_height = int(new_width / original_aspect_ratio)
            if new_height > target_size[1]: new_height = target_size[1]; new_width = int(new_height * original_aspect_ratio)
        else:
            new_height = target_size[1]; new_width = int(new_height * original_aspect_ratio)
            if new_width > target_size[0]: new_width = target_size[0]; new_height = int(new_width / original_aspect_ratio)
        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
        image_array = np.array(image_resized)

        blurred_image = filters.gaussian(image_array, sigma=1)
        thresh_val = filters.threshold_otsu(blurred_image)
        # Heuristic for IVUS: lumen is often dark. Adjust if needed.
        if np.mean(blurred_image[blurred_image > thresh_val]) < np.mean(blurred_image[blurred_image <= thresh_val]):
             binary_image = blurred_image < thresh_val
        else:
             binary_image = blurred_image > thresh_val
        binary_image = binary_fill_holes(binary_image)
        binary_image = morphology.remove_small_objects(binary_image, min_size=100)
        labeled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)
        if num_labels == 0: return None
        region_props = measure.regionprops(labeled_image)
        if not region_props: return None
        largest_region = max(region_props, key=lambda r: r.area)
        lumen_mask = (labeled_image == largest_region.label)
        contours = measure.find_contours(lumen_mask, 0.5)
        if not contours: return None
        contour = sorted(contours, key=len, reverse=True)[0]
        diag = np.sqrt(new_width**2 + new_height**2)
        simplified_contour = measure.approximate_polygon(contour, tolerance=simplify_tolerance * diag)
        if len(simplified_contour) < 3: return None
        normalized_vertices = simplified_contour[:, [1, 0]] / np.array([new_width, new_height])
        if not np.allclose(normalized_vertices[0], normalized_vertices[-1]):
            normalized_vertices = np.vstack([normalized_vertices, normalized_vertices[0]])
        x_coords = normalized_vertices[:, 0]; y_coords = normalized_vertices[:, 1]
        area = 0.5 * np.sum(x_coords * np.roll(y_coords, -1) - np.roll(x_coords, -1) * y_coords)
        if area < 0: normalized_vertices = normalized_vertices[::-1]
        return normalized_vertices
    except Exception as e:
        print(f"Error in extract_lumen_polygon_from_raw_image: {e}"); import traceback; traceback.print_exc(); return None
