import numpy as np
import scipy.io
import trimesh

# Load the Morphable Model
model = scipy.io.loadmat("/Users/edelta076/Desktop/Project_VID_Assistant4/PublicMM1/01_MorphableModel.mat")

# Extract components
shapeMU = model['shapeMU']   # (3N, 1)
shapePC = model['shapePC']   # (3N, n)
shapeEV = model['shapeEV']   # (n, 1)

texMU = model['texMU']       # (3N, 1)
texPC = model['texPC']       # (3N, n)
texEV = model['texEV']       # (n, 1)

tl = model['tl'].T - 1       # Convert to 0-based index for Python

# Number of principal components
n_shape = shapePC.shape[1]
n_tex = texPC.shape[1]

# Generate random coefficients (can use real attributes instead)
alpha = np.random.randn(n_shape, 1) * np.sqrt(shapeEV)
beta = np.random.randn(n_tex, 1) * np.sqrt(texEV)


# Generate shape and texture
shape = shapeMU + shapePC @ alpha  # (3N, 1)
tex = texMU + texPC @ beta         # (3N, 1)

# Reshape
vertices = shape.reshape(-1, 3)    # (N, 3)
colors = tex.reshape(-1, 3) / 255.0  # Normalize RGB to [0,1]

# Create mesh and save
mesh = trimesh.Trimesh(vertices=vertices, faces=tl, vertex_colors=colors)
mesh.export('random_face_mesh_3.ply')

print("Mesh saved as 'random_face_mesh_2.ply'")
