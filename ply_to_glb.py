# import trimesh

# # Load the PLY file
# mesh = trimesh.load("face_mesh_poisson_11zon_resized.ply")

# import numpy as np
# R = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))  # Rotate 180° if needed
# mesh.rotate(R, center=mesh.get_center())

# # Export to GLB
# mesh.export("face_mesh_poisson_11zon_resized.glb")

# mesh = trimesh.load("face_mesh_poisson_11zon_resized.glb")
# mesh.show()

# import trimesh

# # Load the PLY file
# mesh = trimesh.load("face_mesh_poisson_me.ply")

# # Export as glTF (.gltf with external .bin and optional .png)
# mesh.export("face_mesh_poisson_me.gltf")


import trimesh
import numpy as np

# Load the PLY file
mesh = trimesh.load("face_mesh_poisson_fimg1.ply")

# Rotate mesh from Z-up (Open3D) to Y-up (glTF)
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle=np.pi,
    direction=[1, 0, 0],  # rotate around X-axis
    point=mesh.centroid
)
mesh.apply_transform(rotation_matrix)
mesh.show()
# Export to GLB
# mesh.export("face_mesh_poisson_fimg1.glb")



