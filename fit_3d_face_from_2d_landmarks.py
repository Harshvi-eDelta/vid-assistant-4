import numpy as np
import scipy.io
import torchfile
import trimesh
from scipy.optimize import minimize

# ---- STEP 1: Load Morphable Model ----
model = scipy.io.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant4/PublicMM1/01_MorphableModel.mat")
shapeMU = model['shapeMU']  # (3N, 1)
print(shapeMU.shape)
shapePC = model['shapePC']  # (3N, n)
shapeEV = model['shapeEV']  # (n, 1)
tl = model['tl'].T - 1      # (F, 3)

# ---- STEP 2: Load 2D Landmarks ----
t7_landmarks = torchfile.load("/Users/edelta076/Desktop/Project_VID_Assistant4/1.t7")  # change path
landmarks_2d = t7_landmarks[:, :2]  # shape (68, 2)

# ---- STEP 3: Approximate 68 landmark indices on BFM ----
# You MUST replace this with real BFM 68 indices for accurate fitting!
# Below is a rough placeholder assuming face has 53215 vertices

# landmark_indices_3d = np.linspace(1000, 30000, 68).astype(int)

# landmark_indices_3d = np.array([
#     19963, 20205, 21629, 19325, 20983, 20786, 21776, 26140, 43070, 48187,
#     8381, 8374, 23188, 26236, 7441, 8344, 8366, 5392, 8275, 8286,
#     8311, 6389, 8320, 8332, 6912, 5488, 7809, 7032, 7165, 7814,
#     5959, 2088, 4280, 3393, 4158, 40087, 4646, 6796, 4404, 48180,
#     8275, 8286, 8311, 6389, 8320, 8332, 6912, 5488, 7809, 7032,
#     7165, 7814, 5959, 2088, 4280, 3393, 4158, 40087, 4646, 6796,
#     4404, 48180, 8374, 8381, 48187, 23188, 8261, 26236
# ])

landmark_indices_3d = np.array([
    1285, 1280, 1273, 1264, 1252, 1238, 1220, 1201, 1182, 1156, 1129, 1101, 1067, 1032,  991,  951,  911,
    1750, 1763, 1778, 1795, 1810, 1826, 1844,
    1920, 1911, 1900, 1887, 1872, 1857,
    1980, 1995, 2010, 2030, 2048, 2067,
    2080, 2090, 2100, 2110, 2120,
    2155, 2170, 2185, 2200, 2215, 2230, 2245, 2260, 2275, 2290,
    2305, 2320, 2335, 2350, 2365, 2380, 2395, 2410, 2425, 2440,
    2460, 2475, 2490, 2505, 2520, 2535, 2550, 2565
])

landmark_indices_3d = landmark_indices_3d[:68]
print(max(landmark_indices_3d.shape))

# ---- STEP 4: Optimization Function ----
def fit_loss(params):
    n_alpha = shapeEV.shape[0]
    alpha = params[:n_alpha].reshape(-1, 1)
    R = params[n_alpha:n_alpha+3]
    t = params[n_alpha+3:n_alpha+6]
    s = params[n_alpha+6]

    # Rotation matrices for Euler angles
    Rx, Ry, Rz = R
    Rx_mat = np.array([[1, 0, 0],
                       [0, np.cos(Rx), -np.sin(Rx)],
                       [0, np.sin(Rx), np.cos(Rx)]])
    Ry_mat = np.array([[np.cos(Ry), 0, np.sin(Ry)],
                       [0, 1, 0],
                       [-np.sin(Ry), 0, np.cos(Ry)]])
    Rz_mat = np.array([[np.cos(Rz), -np.sin(Rz), 0],
                       [np.sin(Rz), np.cos(Rz), 0],
                       [0, 0, 1]])
    R_mat = Rz_mat @ Ry_mat @ Rx_mat

    # Build shape
    shape = shapeMU + shapePC @ (alpha * shapeEV)
    vertices = shape.reshape(-1, 3)

    # Get 3D landmark points
    landmarks_3d = vertices[landmark_indices_3d]

    # Apply rotation
    rotated = landmarks_3d @ R_mat.T

    # Scaled orthographic projection + translation (only x,y)
    projected = s * rotated[:, :2] + t[:2].reshape(1, 2)

    # Compute MSE loss between projected 2D points and detected landmarks
    return np.mean((projected - landmarks_2d)**2)

# ---- STEP 5: Run Optimization ----
initial_alpha = np.zeros(shapeEV.shape[0])
# initial_params = np.concatenate([initial_alpha, [0, 0, 0], [128, 128]])  # center translation
initial_params = np.concatenate([initial_alpha, [0, 0, 0], [0, 0, 0], [1]])


res = minimize(fit_loss, initial_params, method='L-BFGS-B', options={'maxiter': 200})

# ---- STEP 6: Reconstruct Mesh with Optimal Parameters ----
opt_params = res.x
opt_alpha = opt_params[:shapeEV.shape[0]].reshape(-1, 1)
shape_opt = shapeMU + shapePC @ (opt_alpha * shapeEV)
vertices = shape_opt.reshape(-1, 3)

# ---- STEP 7: Export Mesh ----
mesh = trimesh.Trimesh(vertices=vertices, faces=tl)
mesh.export("fitted_face_2.ply")
print("Saved: fitted_face.ply")

# import torchfile

# data = torchfile.load("/Users/edelta076/Desktop/Project_VID_Assistant4/1.t7")
# print(type(data))
# print(data)

