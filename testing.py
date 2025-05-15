# 3D Face Fitting to 2D Landmarks

import numpy as np
import torchfile
import scipy.io
from scipy.optimize import least_squares
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Load .t7 file
landmarks_t7 = torchfile.load("/Users/edelta076/Desktop/Project_VID_Assistant4/4.t7")
print(landmarks_t7.shape)
landmarks_2d = np.array(landmarks_t7)  # Shape: (68, 2)
print(landmarks_2d.shape)

# Load BFM
bfm = scipy.io.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant4/PublicMM1/01_MorphableModel.mat")

shapeMU = bfm['shapeMU']  # (3*N,1)
shapePC = bfm['shapePC']  # (3*N,K)
shapeEV = bfm['shapeEV']  # (K,1)

N = shapeMU.shape[0] // 3  # Number of vertices

# Load .fp indices (vertex numbers)
def load_fp_indices(fp_path):
    indices = []
    with open(fp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            try:
                idx = int(parts[0])
                indices.append(idx)
            except ValueError:
                continue
    return indices

fp_indices = load_fp_indices("/Users/edelta076/Desktop/Project_VID_Assistant4/bfm_68_landmarks.fp")
print(f'Loaded {len(fp_indices)} landmark indices.')

# Convert indices to zero-based for Python
fp_indices_zero_based = [i-1 for i in fp_indices]  # if indexing starts from 1

# Extract mean shape landmarks (3D points)
landmarks_3d = shapeMU[fp_indices_zero_based, :]  # (68,3) if 68 points

# Dummy detected 2D landmarks - replace with your detected 2D landmarks from your model
# Format: numpy array (68, 2)
landmarks_2d = np.random.rand(len(fp_indices_zero_based), 2) * 256

# Parameters to optimize: shape coefficients + pose (scale, rotation, translation)
# For simplicity, let's optimize shape coeffs only here
K = shapePC.shape[1]
params0 = np.zeros(K)  # Initial shape coefficients

# def project_shape(shape_3d, scale=1.0, translation=np.array([0,0])):
#     """Simple orthographic projection"""
#     projected = shape_3d[:, :2] * scale + translation
#     return projected

# def residuals(params, shapeMU, shapePC, landmark_indices, landmarks_2d):
#     # Compute shape with PCA coefficients
#     shape = shapeMU + shapePC[:, :] @ params  # (3N,)
#     shape = shape.reshape((-1, 3))
#     # Extract only landmarks vertices
#     shape_landmarks = shape[landmark_indices, :]
#     # Project to 2D (using dummy scale and translation for now)
#     shape_proj = project_shape(shape_landmarks, scale=1.0, translation=np.array([0, 0]))
#     return (shape_proj - landmarks_2d).ravel()

# res = least_squares(residuals, params0, args=(shapeMU.reshape(-1), shapePC, fp_indices_zero_based, landmarks_2d))

# print("Optimized shape parameters:", res.x)

def project_shape_with_pose(shape_3d, scale, rot_vec, trans):
    # Apply rotation
    rot = R.from_rotvec(rot_vec)
    rotated = rot.apply(shape_3d)
    # Apply scale and translation
    projected = rotated[:, :2] * scale + trans
    return projected

def residuals_pose(params, shapeMU, shapePC, landmark_indices, landmarks_2d):
    K = shapePC.shape[1]
    shape_coeffs = params[:K]
    scale = params[K]
    rot_vec = params[K+1:K+4]
    trans = params[K+4:K+6]

    shape = shapeMU + shapePC @ shape_coeffs
    shape = shape.reshape((-1, 3))
    shape_landmarks = shape[landmark_indices, :]

    shape_proj = project_shape_with_pose(shape_landmarks, scale, rot_vec, trans)
    return (shape_proj - landmarks_2d).ravel()

# Initial guess: shape zeros, scale=1, no rotation, no translation
params0 = np.zeros(shapePC.shape[1] + 6)
params0[shapePC.shape[1]] = 1.0  # scale initial guess

res = least_squares(residuals_pose, params0,
                    args=(shapeMU.reshape(-1), shapePC, fp_indices_zero_based, landmarks_2d))

print("Optimized shape and pose parameters:", res.x)

# import scipy.io

# # Load the .mat file
# mat_data = scipy.io.loadmat('/Users/edelta076/Desktop/Project_VID_Assistant4/PublicMM1/01_MorphableModel.mat')

# # Print keys to see what variables are inside
# print(mat_data.keys())


