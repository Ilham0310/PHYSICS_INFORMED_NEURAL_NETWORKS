# backend/app.py
import os
os.environ['DDE_BACKEND'] = 'tensorflow' # Force TF2 backend BEFORE all other imports

from flask import Flask, request, jsonify, current_app
import numpy as np
import deepxde as dde
import tensorflow as tf
import sys

PROJECT_ROOT_FOR_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT_FOR_BACKEND)

from piin_model.piin_definition import (
    get_nn_architecture_local,
    get_pde_data_and_net_for_training_local,
    L_CHAR, U_CHAR, RHO_CHAR # Import characteristic scales
)
from backend.utils import extract_lumen_polygon_from_raw_image

app = Flask(__name__)

# --- IMPORTANT: This path MUST MATCH EXACTLY how you save the file in train_and_save_model.py ---
# If your training saves "pinn_model_checkpoint_70000.weights.h5", use that.
# For simplicity, it's often better if train_and_save_model.py saves to a fixed name like "model.weights.h5"
# Let's assume train_and_save_model.py saves to a fixed name for now for robustness.
# If it saves with a step number, you'll need a find_latest_h5_weights_file function.
SAVED_WEIGHTS_FILENAME = "model.weights.h5" # <<--- MAKE SURE THIS MATCHES YOUR TRAINING SCRIPT'S OUTPUT FILENAME
                                           # OR "pinn_model_checkpoint_70000.weights.h5" if that's fixed from last run
MODEL_WEIGHTS_H5_PATH_IN_BACKEND = os.path.join(os.path.dirname(__file__), SAVED_WEIGHTS_FILENAME)


def setup_pinn_for_prediction_local(lumen_polygon_vertices_normalized):
    if not os.path.exists(MODEL_WEIGHTS_H5_PATH_IN_BACKEND):
        msg = f"CRITICAL: Model weights H5 file '{MODEL_WEIGHTS_H5_PATH_IN_BACKEND}' not found. Run training script first."
        print(msg)
        current_app.logger.error(msg)
        return None
    
    print(f"Setting up PINN for prediction with new geometry ({len(lumen_polygon_vertices_normalized)} norm. vertices)...")
    print(f"Attempting to load Keras weights from: {MODEL_WEIGHTS_H5_PATH_IN_BACKEND}")
    
    pde_data_obj = get_pde_data_and_net_for_training_local(
        lumen_polygon_vertices=lumen_polygon_vertices_normalized,
        num_domain=100, num_boundary=50, num_test=50 
    )
    if pde_data_obj is None: current_app.logger.error("pde_data_obj is None"); return None
    net_arch_instance = get_nn_architecture_local()
    if net_arch_instance is None: current_app.logger.error("net_arch_instance is None"); return None

    if dde.backend.backend_name == "tensorflow":
        try:
            dummy_tf_input = tf.zeros((1, 2), dtype=dde.config.real(tf))
            _ = net_arch_instance(dummy_tf_input) # Build the Keras model
            # print("[DEBUG APP] Explicit build call on net_arch_instance successful.")
        except Exception as e_build:
            current_app.logger.error(f"Error explicitly building Keras model: {e_build}", exc_info=True)

    model_for_pred = dde.Model(pde_data_obj, net_arch_instance)
    
    try:
        model_for_pred.compile(optimizer="adam", lr=1e-3, loss="MSE")
        # print("[DEBUG APP] model_for_pred compiled successfully.")
    except Exception as e_compile:
        current_app.logger.error(f"Error during model_for_pred.compile(): {e_compile}", exc_info=True)
        return None

    try:
        # Dummy prediction might still be needed for some Keras initializations
        if len(lumen_polygon_vertices_normalized) > 0:
            dummy_center_x_star = np.mean(lumen_polygon_vertices_normalized[:,0])
            dummy_center_y_star = np.mean(lumen_polygon_vertices_normalized[:,1])
        else: dummy_center_x_star, dummy_center_y_star = 0.5, 0.5 # Should not happen
        dummy_predict_input_np = np.array([[dummy_center_x_star, dummy_center_y_star]], dtype=dde.config.real(np))
        _ = model_for_pred.predict(dummy_predict_input_np)
        # print("[DEBUG APP] Dummy prediction successful.")
        
        # Load Keras weights directly into the network object (model_for_pred.net)
        model_for_pred.net.load_weights(MODEL_WEIGHTS_H5_PATH_IN_BACKEND)
        print(f"Keras model weights loaded successfully from {MODEL_WEIGHTS_H5_PATH_IN_BACKEND}")
        return model_for_pred
    except Exception as e:
        current_app.logger.error(f"Error in setup_pinn_for_prediction_local (load_weights or predict): {e}", exc_info=True)
        return None

# PASTE THE REST OF YOUR @app.route('/predict') and if __name__ == '__main__': block here
# from the version in response ID: gcM259p6bB0xJqO3i4_fNA (the previous "complete app.py")
# The changes are only within setup_pinn_for_prediction_local and MODEL_WEIGHTS_H5_PATH_IN_BACKEND

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files: return jsonify({"error": "No image file"}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        lumen_vertices_normalized = extract_lumen_polygon_from_raw_image(image_bytes)
        if lumen_vertices_normalized is None or len(lumen_vertices_normalized) < 3:
            return jsonify({"error": "Failed to segment lumen or polygon too small."}), 500

        pinn_model_instance = setup_pinn_for_prediction_local(lumen_vertices_normalized)
        if pinn_model_instance is None: return jsonify({"error": "Failed to initialize PINN model."}), 500

        geom_obj_pred = pinn_model_instance.data.geom
        num_pred_points = 500
        pred_points_star_np = geom_obj_pred.random_points(num_pred_points).astype(dde.config.real(np))
        predictions_star_raw = pinn_model_instance.predict(pred_points_star_np)
        
        u_star_preds = predictions_star_raw[:, 0]
        v_star_preds = predictions_star_raw[:, 1]
        p_star_preds = predictions_star_raw[:, 2]

        u_physical_preds = u_star_preds * U_CHAR
        v_physical_preds = v_star_preds * U_CHAR
        p_physical_preds = p_star_preds * (RHO_CHAR * U_CHAR**2)

        valid_mask = ~np.isnan(p_physical_preds) & ~np.isinf(p_physical_preds) # Check one, assume others similar
        
        avg_u_physical, avg_v_physical, avg_p_physical = 0.0, 0.0, 0.0
        if np.any(valid_mask): # Check if there are any valid predictions
            avg_u_physical = float(np.mean(u_physical_preds[valid_mask]))
            avg_v_physical = float(np.mean(v_physical_preds[valid_mask]))
            avg_p_physical = float(np.mean(p_physical_preds[valid_mask]))
        else:
            current_app.logger.warning("No valid (non-nan/inf) physical predictions generated for averaging.")

        
        centroid_x_star = float(np.mean(lumen_vertices_normalized[:, 0])) if len(lumen_vertices_normalized) > 0 else 0.5
        centroid_y_star = float(np.mean(lumen_vertices_normalized[:, 1])) if len(lumen_vertices_normalized) > 0 else 0.5
        center_pred_u_physical, center_pred_v_physical, center_pred_p_physical = 0.0, 0.0, 0.0
        
        centroid_point_star_np = np.array([[centroid_x_star, centroid_y_star]], dtype=dde.config.real(np))

        # Check if centroid is "inside" according to your STUB method
        # geom_obj_pred.inside expects a 2D array of points, so check inside_result[0]
        is_inside = False
        if hasattr(geom_obj_pred, 'inside') and callable(geom_obj_pred.inside):
            inside_result = geom_obj_pred.inside(centroid_point_star_np)
            if isinstance(inside_result, (np.ndarray, tf.Tensor)) and inside_result.size > 0:
                is_inside = bool(inside_result[0]) # Get the boolean value for the first (only) point

        if is_inside:
            center_pred_star_list = pinn_model_instance.predict(centroid_point_star_np)
            if center_pred_star_list is not None and len(center_pred_star_list) > 0:
                center_pred_star = center_pred_star_list[0] # predict returns a list of arrays for each output (usually one for FNN)
                if center_pred_star.ndim > 1: # If it's like [[u,v,p]]
                    center_pred_star = center_pred_star[0]

                u_star_center, v_star_center, p_star_center = center_pred_star[0], center_pred_star[1], center_pred_star[2]
                center_pred_u_physical = float(u_star_center * U_CHAR) if not np.isnan(u_star_center) else 0.0
                center_pred_v_physical = float(v_star_center * U_CHAR) if not np.isnan(v_star_center) else 0.0
                center_pred_p_physical = float(p_star_center * (RHO_CHAR * U_CHAR**2)) if not np.isnan(p_star_center) else 0.0
            else:
                 current_app.logger.warning("Prediction at centroid returned None or empty list.")
        else:
            current_app.logger.info("Lumen centroid is outside the (STUB) 'inside' definition of the geometry. Center prediction skipped.")

        
        return jsonify({
            "message": "Prediction successful", "num_lumen_vertices": len(lumen_vertices_normalized),
            "lumen_centroid_normalized": {"x_star": centroid_x_star, "y_star": centroid_y_star},
            "center_point_prediction_physical": {"u_velocity_mps": center_pred_u_physical, "v_velocity_mps": center_pred_v_physical, "pressure_Pa": center_pred_p_physical},
            "average_values_in_domain_physical": {"avg_u_velocity_mps": avg_u_physical, "avg_v_velocity_mps": avg_v_physical, "avg_pressure_Pa": avg_p_physical},
            "debug_lumen_vertices_normalized": lumen_vertices_normalized.tolist(),
            "debug_characteristic_scales": {"L_char_m": L_CHAR, "U_char_mps": U_CHAR, "Rho_char_kgm3": RHO_CHAR}
        })
    except Exception as e:
        current_app.logger.error(f"Unhandled exception in /predict: {e}", exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    if not app.debug:
        import logging
        logging.basicConfig(level=logging.INFO)
    if not os.path.exists(MODEL_WEIGHTS_H5_PATH_IN_BACKEND):
        app.logger.warning("*"*50 + f"\nWARNING: {os.path.basename(MODEL_WEIGHTS_H5_PATH_IN_BACKEND)} not found.\n" + "*"*50)
    app.run(host='0.0.0.0', port=5000, debug=True)
