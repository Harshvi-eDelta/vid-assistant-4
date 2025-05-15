import numpy as np
import trimesh
import scipy.io

# Load BFM model
model = scipy.io.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant4/PublicMM1/01_MorphableModel.mat")

# Extract necessary arrays from the model
shapeMU = model['shapeMU']        # Mean shape (3N x 1)
shapePC = model['shapePC']        # Shape principal components (3N x M)
shapeEV = model['shapeEV'].flatten()  # Eigenvalues (M,)
tl = model['tl'].T - 1            # Triangle list, convert 1-based MATLAB index to 0-based

# Number of shape components to use (tune this, max ~199)
num_components = 80

# Generate random shape coefficients (identity parameters)
shape_coeffs = np.random.randn(num_components) * 1.0  # You can adjust multiplier for variation

# Compute the shape variation
weighted_shape = shapePC[:, :num_components] @ (shape_coeffs * shapeEV[:num_components])

# Add variation to mean shape
detailed_shape = shapeMU.flatten() + weighted_shape

# Reshape to (num_vertices, 3) with correct order
num_vertices = detailed_shape.shape[0] // 3
vertices = detailed_shape.reshape((3, num_vertices), order='F').T  # (N, 3)

# Create trimesh object
mesh = trimesh.Trimesh(vertices=vertices, faces=tl.astype(np.int32), process=False)

# Export to .ply or .obj for viewing
mesh.export('bfm_detailed_face.ply')


# to check .mat files key
'''import scipy.io as sio

bfm = sio.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant4/PublicMM1/01_MorphableModel.mat")
print(bfm.keys())'''
