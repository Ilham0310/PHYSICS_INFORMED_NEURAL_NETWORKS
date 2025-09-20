# pinn_blood_flow_local/pinn_model/pinn_definition.py
import deepxde as dde
import numpy as np
import tensorflow as tf
import sys

dde.config.set_default_float("float32")
print(f"DeepXDE default float type set to: {dde.config.default_float()}")

# --- Characteristic Physical Scales (Example Values in SI Units) ---
L_CHAR = 0.003  # meters (e.g., 3mm lumen diameter)
U_CHAR = 0.3    # m/s (e.g., 30 cm/s characteristic inlet velocity)
RHO_CHAR = 1060 # kg/m^3 (density of blood)
MU_CHAR = 0.0035 # Pa.s (dynamic viscosity of blood)
NU_CHAR = MU_CHAR / RHO_CHAR # m^2/s
REYNOLDS_NUMBER = (U_CHAR * L_CHAR) / NU_CHAR
# print(f"[Physics Config] Characteristic Reynolds Number (Re): {REYNOLDS_NUMBER:.2f}")

RHO_PINN_ND = 1.0
NU_PINN_ND_EFF = (1.0 / REYNOLDS_NUMBER) if REYNOLDS_NUMBER > 1e-9 else 1e6
# print(f"[Physics Config] PINN RHO_PARAM (non-dim rho): {RHO_PINN_ND}")
# print(f"[Physics Config] PINN NU_PARAM (non-dim nu, effectively 1/Re): {NU_PINN_ND_EFF:.3e}")

def navier_stokes_2d_local(x_in, y_out):
    u_star = y_out[:, 0:1]
    v_star = y_out[:, 1:2]
    p_star = y_out[:, 2:3]
    rho_nd = tf.constant(RHO_PINN_ND, dtype=tf.float32)
    nu_nd_eff  = tf.constant(NU_PINN_ND_EFF, dtype=tf.float32)
    inv_rho_nd = tf.constant(1.0, dtype=tf.float32) / rho_nd
    du_star_dx_star = dde.grad.jacobian(y_out, x_in, i=0, j=0)
    du_star_dy_star = dde.grad.jacobian(y_out, x_in, i=0, j=1)
    dv_star_dx_star = dde.grad.jacobian(y_out, x_in, i=1, j=0)
    dv_star_dy_star = dde.grad.jacobian(y_out, x_in, i=1, j=1)
    dp_star_dx_star = dde.grad.jacobian(y_out, x_in, i=2, j=0)
    dp_star_dy_star = dde.grad.jacobian(y_out, x_in, i=2, j=1)
    du_star_dxx_star = dde.grad.hessian(y_out, x_in, component=0, i=0, j=0)
    du_star_dyy_star = dde.grad.hessian(y_out, x_in, component=0, i=1, j=1)
    dv_star_dxx_star = dde.grad.hessian(y_out, x_in, component=1, i=0, j=0)
    dv_star_dyy_star = dde.grad.hessian(y_out, x_in, component=1, i=1, j=1)
    continuity_nd = du_star_dx_star + dv_star_dy_star
    x_momentum_nd = (u_star * du_star_dx_star + v_star * du_star_dy_star +
                     inv_rho_nd * dp_star_dx_star -
                     nu_nd_eff * (du_star_dxx_star + du_star_dyy_star))
    y_momentum_nd = (u_star * dv_star_dx_star + v_star * dv_star_dy_star +
                     inv_rho_nd * dp_star_dy_star -
                     nu_nd_eff * (dv_star_dxx_star + dv_star_dyy_star))
    return [continuity_nd, x_momentum_nd, y_momentum_nd]

class PolygonGeometryLocal(dde.geometry.Geometry):
    def __init__(self, vertices):
        self.vertices_orig = np.array(vertices, dtype=np.float32) # Keep original for reference
        if self.vertices_orig.shape[0] < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        
        # Ensure polygon is closed for edge calculations
        if not np.allclose(self.vertices_orig[0], self.vertices_orig[-1]):
            self.vertices = np.vstack([self.vertices_orig, self.vertices_orig[0]])
        else:
            self.vertices = self.vertices_orig # Already closed
            
        self.xmin, self.xmax = self.vertices[:, 0].min(), self.vertices[:, 0].max()
        self.ymin, self.ymax = self.vertices[:, 1].min(), self.vertices[:, 1].max()
        diam_float64 = np.hypot(self.xmax - self.xmin, self.ymax - self.ymin)
        
        # Check for degenerate bounding box immediately
        if np.isclose(self.xmin, self.xmax, atol=1e-6) or np.isclose(self.ymin, self.ymax, atol=1e-6):
            print(f"[WARN PolyGeo Init] Degenerate bounding box: x:[{self.xmin:.2e},{self.xmax:.2e}], y:[{self.ymin:.2e},{self.ymax:.2e}]")
            # This geometry is problematic for sampling from bounding box.
            # Forcing diam to be small but non-zero to avoid issues in super().__init__
            diam_float64 = max(diam_float64, 1e-6)


        super().__init__(2, 
                         (float(self.xmin), float(self.ymin), float(self.xmax), float(self.ymax)), 
                         float(diam_float64))

        # Area calculation using unique vertices (if polygon was closed by duplicating)
        unique_verts_for_area = self.vertices[:-1] if np.allclose(self.vertices[0], self.vertices[-1]) else self.vertices
        if unique_verts_for_area.shape[0] >=3:
            x_coords = unique_verts_for_area[:, 0] 
            y_coords = unique_verts_for_area[:, 1]
            self.area_val = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - 
                                        np.dot(y_coords, np.roll(x_coords, 1)))
        else:
            self.area_val = 0.0
        # print(f"[DEBUG PolyGeo] Initialized. Area: {self.area_val:.4e}. BBox: x[{self.xmin:.2e},{self.xmax:.2e}], y[{self.ymin:.2e},{self.ymax:.2e}]")

    def inside(self, x): # STUB: Bounding box check
        return ((x[:, 0] >= self.xmin) & (x[:, 0] <= self.xmax) &
                (x[:, 1] >= self.ymin) & (x[:, 1] <= self.ymax))

    def on_boundary(self, x): # STUB: Bounding box edge check
        tol = np.float32(1e-3)
        on_xmin = np.isclose(x[:, 0], self.xmin, atol=tol); on_xmax = np.isclose(x[:, 0], self.xmax, atol=tol)
        on_ymin = np.isclose(x[:, 1], self.ymin, atol=tol); on_ymax = np.isclose(x[:, 1], self.ymax, atol=tol)
        return (on_xmin | on_xmax | on_ymin | on_ymax) & self.inside(x)

    def random_points(self, n, random="pseudo"): # STUB: Samples from bounding box
        if np.isclose(self.xmin, self.xmax, atol=1e-6) or np.isclose(self.ymin, self.ymax, atol=1e-6):
             # print(f"[WARN PolyGeo random_points] Degenerate bbox, returning empty for domain points.")
             return np.empty((0, self.dim), dtype=np.float32)
        x_coords = np.random.uniform(self.xmin, self.xmax, n).astype(np.float32)
        y_coords = np.random.uniform(self.ymin, self.ymax, n).astype(np.float32)
        return np.vstack((x_coords, y_coords)).T

    def random_boundary_points(self, n, random="pseudo"):
        num_verts = len(self.vertices) # self.vertices is now guaranteed closed
        if num_verts <= 1: return np.empty((0, self.dim), dtype=np.float32) # Not enough for segments

        edge_vectors = np.diff(self.vertices, axis=0) # Vectors from v_i to v_{i+1} for a closed loop
        edge_lengths = np.sqrt(np.sum(edge_vectors**2, axis=1))
        
        total_length = np.sum(edge_lengths)
        if total_length < 1e-7: return np.empty((0, self.dim), dtype=np.float32)

        cumulative_lengths = np.cumsum(edge_lengths)
        points_on_boundary = np.empty((n, self.dim), dtype=np.float32)
        random_perimeter_distances = np.random.uniform(0, total_length, n)

        for i in range(n):
            dist = random_perimeter_distances[i]
            edge_idx = np.searchsorted(cumulative_lengths, dist)
            if edge_idx >= len(edge_lengths): edge_idx = len(edge_lengths) - 1 # Should not happen if total_length > 0

            v1 = self.vertices[edge_idx]
            v2 = self.vertices[edge_idx + 1] # Accesses the closing vertex (which is self.vertices[0])
            
            dist_along_segment = dist - (cumulative_lengths[edge_idx - 1] if edge_idx > 0 else 0.0)
            segment_len_curr = edge_lengths[edge_idx]
            
            t = (dist_along_segment / segment_len_curr) if segment_len_curr > 1e-9 else 0.0
            points_on_boundary[i] = v1 * (1.0 - t) + v2 * t
        return points_on_boundary

CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL = None

def _calculate_robust_tolerance(min_val, max_val, relative_factor=0.05, absolute_min_tol=1e-4, range_atol=1e-5):
    """Helper to calculate a robust tolerance for BC predicates."""
    effective_range = max_val - min_val
    if np.isclose(float(effective_range), 0.0, atol=range_atol): # If range is effectively zero
        # Use a small fraction of a typical normalized dimension (e.g., 1.0)
        # or an absolute minimum based on expected precision needs.
        tol = np.float32(0.01) # Default for zero range, e.g. 1% of a unit normalized dimension
    else:
        tol = np.float32(relative_factor) * effective_range
    return max(tol, np.float32(absolute_min_tol)) # Ensure a minimum absolute tolerance

def boundary_inlet_local_fn(x, on_boundary):
    global CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL
    if CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL is None or not on_boundary: return False
    min_x_star = CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL[:, 0].min()
    max_x_star = CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL[:, 0].max() # Needed for range
    tol_star = _calculate_robust_tolerance(min_x_star, max_x_star)
    return np.isclose(x[0], min_x_star, atol=tol_star)

def boundary_outlet_local_fn(x, on_boundary):
    global CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL
    if CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL is None or not on_boundary: return False
    min_x_star = CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL[:, 0].min() # Needed for range
    max_x_star = CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL[:, 0].max()
    tol_star = _calculate_robust_tolerance(min_x_star, max_x_star)
    return np.isclose(x[0], max_x_star, atol=tol_star)

def boundary_wall_local_fn(x, on_boundary):
    return on_boundary and not (boundary_inlet_local_fn(x, True) or boundary_outlet_local_fn(x, True))

def u_star_inlet_val(x):
    return tf.constant(1.0, dtype=tf.float32)

def get_bcs_for_geom_local(geom_obj):
    global CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL
    if hasattr(geom_obj, "vertices"): CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL = geom_obj.vertices
    else: CURRENT_LUMEN_VERTICES_FOR_BC_LOCAL = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    
    zero_fn_star = lambda x_val: tf.zeros_like(x_val[:, 0:1], dtype=tf.float32)
    bc_u_star_inlet = dde.icbc.DirichletBC(geom_obj, u_star_inlet_val, boundary_inlet_local_fn, component=0)
    bc_v_star_inlet = dde.icbc.DirichletBC(geom_obj, zero_fn_star, boundary_inlet_local_fn, component=1)
    bc_u_star_wall = dde.icbc.DirichletBC(geom_obj, zero_fn_star, boundary_wall_local_fn, component=0)
    bc_v_star_wall = dde.icbc.DirichletBC(geom_obj, zero_fn_star, boundary_wall_local_fn, component=1)
    bc_p_star_outlet = dde.icbc.DirichletBC(geom_obj, zero_fn_star, boundary_outlet_local_fn, component=2)
    return [bc_u_star_inlet, bc_v_star_inlet, bc_u_star_wall, bc_v_star_wall, bc_p_star_outlet]

def get_nn_architecture_local():
    return dde.nn.FNN([2] + [64] * 4 + [3], "tanh", "Glorot normal")

def get_pde_data_and_net_for_training_local(
    lumen_polygon_vertices, num_domain=1000, num_boundary=500, num_test=500
):
    if lumen_polygon_vertices is not None and len(lumen_polygon_vertices) >= 3:
        current_geom = PolygonGeometryLocal(lumen_polygon_vertices)
        # Check if geometry became degenerate after init (e.g. area too small)
        if current_geom.area_val < 1e-7 or \
           np.isclose(current_geom.xmin, current_geom.xmax, atol=1e-6) or \
           np.isclose(current_geom.ymin, current_geom.ymax, atol=1e-6):
            print(f"Warning: Polygon resulted in degenerate geometry object. Area: {current_geom.area_val:.2e}. Using default Rectangle.")
            current_geom = dde.geometry.Rectangle([0,0], [1,1])
    else:
        current_geom = dde.geometry.Rectangle([0,0], [1,1])
        if lumen_polygon_vertices is not None: # It was not None, but had < 3 vertices
             print(f"Warning: Lumen polygon vertices had < 3 points ({len(lumen_polygon_vertices)}). Using default Rectangle.")
        else: # It was None
            print(f"Warning: Lumen polygon vertices was None. Using default Rectangle.")
            
    boundary_conditions = get_bcs_for_geom_local(current_geom)
    pde_data = dde.data.PDE(
        current_geom, navier_stokes_2d_local, boundary_conditions,
        num_domain=num_domain, num_boundary=num_boundary, num_test=num_test,
    )
    return pde_data
