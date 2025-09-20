# pinn_blood_flow_local/pinn_model/train_and_save_model.py
import os
# Force backend BEFORE importing deepxde or tensorflow
os.environ['DDE_BACKEND'] = 'tensorflow'

import deepxde as dde
import numpy as np
import random
import tensorflow as tf # Keep this import, especially if using tf.keras.optimizers.Adam
from tensorflow.keras.optimizers import Adam # Import Adam specifically for gradient clipping

from piin_model.dataset_utils import (
    get_processed_image_mask_pairs,
    extract_polygon_from_mask_file
)
from piin_model.piin_definition import (
    get_nn_architecture_local,
    get_pde_data_and_net_for_training_local
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
os.makedirs(BACKEND_DIR, exist_ok=True)
OUTPUT_MODEL_CHECKPOINT_PREFIX = os.path.join(BACKEND_DIR, "pinn_model_checkpoint")

# Hyperparameters
ITERATIONS_PER_GEOMETRY_LOCAL = 1000
LEARNING_RATE_LOCAL = 1e-4
MAX_SAMPLES_TO_TRAIN_ON_LOCAL = None # Start with a small number for testing
TRAINING_SPLIT_TO_USE = "train"
GRADIENT_CLIPNORM = 1.0

def run_local_pinn_training_from_processed_files():
    print(f"--- Starting Local PINN Training ---")
    # Verify backend and float type again at the start of actual execution
    print(f"DeepXDE Backend in use: {dde.backend.backend_name}")
    print(f"DeepXDE Default Float: {dde.config.default_float()}")

    print(f"Model checkpoint prefix: {OUTPUT_MODEL_CHECKPOINT_PREFIX}")
    print(f"Iterations per geometry: {ITERATIONS_PER_GEOMETRY_LOCAL}")
    print(f"Learning rate: {LEARNING_RATE_LOCAL}")
    print(f"Gradient clipnorm: {GRADIENT_CLIPNORM}")
    
    image_mask_pairs = get_processed_image_mask_pairs(split_name=TRAINING_SPLIT_TO_USE)
    if not image_mask_pairs: 
        print(f"No pairs found for split '{TRAINING_SPLIT_TO_USE}'. Aborting.")
        return

    training_samples = image_mask_pairs
    if MAX_SAMPLES_TO_TRAIN_ON_LOCAL is not None and 0 < MAX_SAMPLES_TO_TRAIN_ON_LOCAL < len(image_mask_pairs):
        print(f"Limiting training to {MAX_SAMPLES_TO_TRAIN_ON_LOCAL} random samples.")
        training_samples = random.sample(image_mask_pairs, MAX_SAMPLES_TO_TRAIN_ON_LOCAL)
    
    if not training_samples: 
        print("No samples selected for training. Aborting.")
        return

    shared_neural_network = get_nn_architecture_local()
    print("\n--- Initializing shared network and optimizer ---")
    
    first_valid_lumen_vertices = None
    processed_mask_for_init = "N/A"
    for pair_data in training_samples: 
        poly_vertices = extract_polygon_from_mask_file(pair_data['mask_path'])
        if poly_vertices is not None and len(poly_vertices) >= 3:
            first_valid_lumen_vertices = poly_vertices
            processed_mask_for_init = os.path.basename(pair_data['mask_path'])
            print(f"Found first valid geometry for initialization: {processed_mask_for_init}.")
            break
    
    if first_valid_lumen_vertices is None: 
        print("ERROR: No valid lumen polygon found in the selected samples. Aborting.")
        return
    
    print(f"\n[DEBUG train_script] Initializing with 'first_valid_lumen_vertices' (Shape: {first_valid_lumen_vertices.shape}, Dtype: {first_valid_lumen_vertices.dtype})")
    # ... (optional detailed print of ranges for first_valid_lumen_vertices) ...

    initial_pde_data_obj = get_pde_data_and_net_for_training_local(
        lumen_polygon_vertices=first_valid_lumen_vertices,
        num_domain=100, num_boundary=50, num_test=50
    )
    pinn_training_model = dde.Model(initial_pde_data_obj, shared_neural_network)
    
    optimizer_instance = Adam(learning_rate=LEARNING_RATE_LOCAL, clipnorm=GRADIENT_CLIPNORM)
    # For DeepXDE, when passing an optimizer object, you often don't specify lr again in compile
    pinn_training_model.compile(optimizer=optimizer_instance) 
    print("Shared neural network and Adam optimizer initialized (with gradient clipping).\n")
    
    # Verify clipnorm (optional, for peace of mind)
    # Actual Keras optimizer is often at self.opt (or self.optimizer) after compile
    if hasattr(pinn_training_model, 'opt') and hasattr(pinn_training_model.opt, 'clipnorm'):
         print(f"Gradient clipping is active with clipnorm = {pinn_training_model.opt.clipnorm.numpy() if hasattr(pinn_training_model.opt.clipnorm, 'numpy') else pinn_training_model.opt.clipnorm}")
    elif hasattr(pinn_training_model, 'optimizer') and hasattr(pinn_training_model.optimizer, 'clipnorm'):
         print(f"Gradient clipping is active with clipnorm = {pinn_training_model.optimizer.clipnorm.numpy() if hasattr(pinn_training_model.optimizer.clipnorm, 'numpy') else pinn_training_model.optimizer.clipnorm}")
    else:
        print("Note: Could not directly verify clipnorm on optimizer post-compile via standard attributes. Assuming it's set if Adam object was passed.")


    num_samples_trained = 0
    total_iterations_tracker = 0 

    for sample_idx, current_pair_data in enumerate(training_samples):
        print(f"--- Processing Sample {sample_idx + 1}/{len(training_samples)}: {os.path.basename(current_pair_data['mask_path'])} ---")
        current_lumen_poly_vertices = extract_polygon_from_mask_file(current_pair_data['mask_path'])

        if current_lumen_poly_vertices is None or len(current_lumen_poly_vertices) < 3:
            print(f"Skipping sample: Invalid polygon extracted.")
            continue
        
        # Optional Debug print for polygon properties
        min_x_s = np.min(current_lumen_poly_vertices[:,0]); max_x_s = np.max(current_lumen_poly_vertices[:,0])
        eff_r_s = max_x_s - min_x_s
        min_y_s = np.min(current_lumen_poly_vertices[:,1]); max_y_s = np.max(current_lumen_poly_vertices[:,1])
        eff_r_y_s = max_y_s - min_y_s
        # print(f"  Current Poly x* range: [{min_x_s:.4e}, {max_x_s:.4e}], eff_range_x_star: {eff_r_s:.4e}")
        # print(f"  Current Poly y* range: [{min_y_s:.4e}, {max_y_s:.4e}], eff_range_y_star: {eff_r_y_s:.4e}")
        if eff_r_s < 1e-3 or eff_r_y_s < 1e-3:
             print(f"  [WARN Train] Current polygon for sample {sample_idx+1} has very small x or y range. May be challenging.")

        num_samples_trained += 1
        current_pde_data_obj = get_pde_data_and_net_for_training_local(
            lumen_polygon_vertices=current_lumen_poly_vertices,
            num_domain=1500, num_boundary=1000, num_test=500
        )
        pinn_training_model.data = current_pde_data_obj
        
        # Re-compiling with the same optimizer instance should be okay and ensures
        # the optimizer is correctly associated with the new data if necessary.
        # However, simply updating model.data might be sufficient if Adam adapts well.
        # For safety, can re-compile:
        pinn_training_model.compile(optimizer=optimizer_instance)

        print(f"Training for {ITERATIONS_PER_GEOMETRY_LOCAL} iterations on current geometry...")
        loss_history, train_state = pinn_training_model.train(
            iterations=ITERATIONS_PER_GEOMETRY_LOCAL, 
            display_every=max(100, ITERATIONS_PER_GEOMETRY_LOCAL // 10)
        )
        
        if train_state is not None:
            total_iterations_tracker = train_state.best_step 
            
            # --- MODIFIED LOSS ACCESS ---
            # Check for 'best_loss_train' (often used by DDE for the value associated with best_step)
            # or 'best_loss_test' or fallback to current 'loss_train'
            final_loss_for_this_geom = float('nan') # Default to nan
            if hasattr(train_state, 'best_loss_test') and train_state.best_loss_test is not None:
                final_loss_for_this_geom = train_state.best_loss_test
            elif hasattr(train_state, 'best_loss_train') and train_state.best_loss_train is not None:
                 final_loss_for_this_geom = train_state.best_loss_train
            elif hasattr(train_state, 'loss_train') and train_state.loss_train is not None: # loss_train is usually a list
                # Sum of current train losses if 'best_loss' attributes are not present
                if isinstance(train_state.loss_train, (list, tuple)) and all(isinstance(item, (int, float)) for item in train_state.loss_train):
                    final_loss_for_this_geom = sum(train_state.loss_train) 
                elif isinstance(train_state.loss_train, (int,float)): # if it's a single value
                    final_loss_for_this_geom = train_state.loss_train

            print(f"  Finished training for this geometry. Best step reported for this geo: {train_state.best_step}, Associated loss: {final_loss_for_this_geom:.3e}")
            # --- END MODIFIED LOSS ACCESS ---

            if np.any(np.isnan(final_loss_for_this_geom)) or (hasattr(train_state, 'loss_train') and np.any(np.isnan(train_state.loss_train))):
                print(f"  [CRITICAL WARN] NaN loss detected for sample {sample_idx+1}. Training may be unstable.")
        else:
            total_iterations_tracker += ITERATIONS_PER_GEOMETRY_LOCAL 
            print("  [WARN Train] train_state was None after training an epoch.")


    if num_samples_trained == 0: 
        print(f"No valid samples processed. Model not saved.")
        return

    try:
        # DeepXDE's model.save uses the model's internal global step,
        # which is tracked by best_step from the last train_state if checkpointing is enabled,
        # or its own internal step counter.
        # Providing total_iterations_tracker to save method can be version-dependent.
        # Usually, just providing the prefix is enough.
        saved_path_prefix = pinn_training_model.save(OUTPUT_MODEL_CHECKPOINT_PREFIX, verbose=1)
        
        print(f"\n--- Training Complete ---")
        print(f"DeepXDE Model Checkpoint saved with actual prefix: {saved_path_prefix}")
        print(f"Trained on {num_samples_trained} geometries from split '{TRAINING_SPLIT_TO_USE}'.")
    except Exception as e:
        print(f"ERROR saving final model checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure TensorFlow is imported if not already, for Adam optimizer.
    # This might not be strictly necessary if dde handles it, but good for clarity.
    # import tensorflow as tf 
    
    # Print backend being used by DeepXDE at the very start of script execution
    # Note: os.environ must be set *before* first import of deepxde
    print(f"Script starting. DeepXDE Backend (will be): {os.environ.get('DDE_BACKEND', 'Default (likely tensorflow if available)')}")
    print(f"TensorFlow version: {tf.__version__}")
    
    run_local_pinn_training_from_processed_files()
