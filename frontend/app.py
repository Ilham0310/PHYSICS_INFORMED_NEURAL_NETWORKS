# pinn_blood_flow_local/frontend/app.py
import streamlit as st
import requests
from PIL import Image
import io
import json
import numpy as np
import matplotlib.pyplot as plt

BACKEND_URL = "http://localhost:5000/predict"

st.set_page_config(layout="wide")
st.title("🩸 PINN-based Blood Flow Simulation (IVUS Geometry)")

st.markdown("""
Upload an IVUS-like image. The system will attempt to segment the lumen,
define it as a 2D polygonal domain, and use a pre-trained PINN
to predict conceptual blood flow parameters (pressure, velocity) in physical units.
""")
st.warning("""
**Disclaimer:** This is a demonstration.
- The IVUS image segmentation is basic.
- The PINN's geometry handling for arbitrary shapes still uses STUBs for `inside()` and domain point sampling.
- Boundary conditions are simplified.
**Accuracy for clinical use is NOT implied.**
""")

uploaded_file = st.file_uploader("Upload IVUS Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file) 
    col1, col2 = st.columns([1, 2]) 
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_pil, caption="Uploaded IVUS-like Image", use_column_width=True)

    if st.button("🚀 Predict Blood Flow Parameters", key="predict_button"):
        with st.spinner("Processing IVUS image and running PINN simulation..."):
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(BACKEND_URL, files=files, timeout=90) 
                response.raise_for_status()
                results = response.json()

                with col2:
                    st.subheader("📊 Prediction Results")
                    if "error" in results:
                        st.error(f"Error from backend: {results['error']}")
                    else:
                        st.success(results.get("message", "Prediction successful!"))
                        st.write(f"**Lumen Polygon Vertices Found:** {results.get('num_lumen_vertices', 'N/A')}")
                        
                        centroid_norm = results.get("lumen_centroid_normalized", {})
                        st.write(f"**Approx. Lumen Centroid (Normalized Coords):** "
                                 f"x*={centroid_norm.get('x_star', 0):.3f}, y*={centroid_norm.get('y_star', 0):.3f}")
                        
                        st.markdown("#### Center Point Prediction (Physical Units)")
                        center_pred_phys = results.get("center_point_prediction_physical", {})
                        st.metric(label="Pressure (Pa)", value=f"{center_pred_phys.get('pressure_Pa', 0):.4f}")
                        st.metric(label="U-Velocity (m/s)", value=f"{center_pred_phys.get('u_velocity_mps', 0):.4f}")
                        st.metric(label="V-Velocity (m/s)", value=f"{center_pred_phys.get('v_velocity_mps', 0):.4f}")
                        
                        st.markdown("#### Average Values in Domain (Physical Units, Approx. over BBox)")
                        avg_vals_phys = results.get("average_values_in_domain_physical", {})
                        st.metric(label="Average Pressure (Pa)", value=f"{avg_vals_phys.get('avg_pressure_Pa', 0):.4f}")
                        st.metric(label="Average U-Velocity (m/s)", value=f"{avg_vals_phys.get('avg_u_velocity_mps', 0):.4f}")
                        st.metric(label="Average V-Velocity (m/s)", value=f"{avg_vals_phys.get('avg_v_velocity_mps', 0):.4f}")
                        
                        if "debug_characteristic_scales" in results:
                            st.markdown("---")
                            st.markdown("##### Characteristic Scales Used:")
                            scales = results["debug_characteristic_scales"]
                            st.json(scales) # Display the scales used for conversion

                        if st.checkbox("Show Segmented Lumen Polygon (Normalized)", value=False):
                            lumen_verts_norm = results.get("debug_lumen_vertices_normalized")
                            if lumen_verts_norm:
                                verts_np = np.array(lumen_verts_norm)
                                fig, ax = plt.subplots()
                                ax.plot(verts_np[:, 0], verts_np[:, 1], 'r-') # Plot x*, y*
                                ax.fill(verts_np[:, 0], verts_np[:, 1], 'r', alpha=0.3)
                                ax.set_title("Segmented Lumen (Normalized Coordinates [0,1])")
                                ax.set_xlabel("X* (Normalized)"); ax.set_ylabel("Y* (Normalized)")
                                ax.set_aspect('equal', adjustable='box'); ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
                                st.pyplot(fig)
            except requests.exceptions.RequestException as e:
                with col2: st.error(f"Request to backend failed: {e}")
            except json.JSONDecodeError:
                with col2: st.error(f"Failed to decode JSON from backend. Raw: {response.text[:500]}...")
            except Exception as e:
                 with col2: st.error(f"Frontend error: {e}"); import traceback; st.text(traceback.format_exc())
else:
    st.info("Please upload an image.")

st.sidebar.header("About")
st.sidebar.info("PINN Demo for IVUS Blood Flow (Non-Dimensionalized).")
