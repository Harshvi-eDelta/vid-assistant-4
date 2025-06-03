import trimesh

# Load the PLY file
mesh = trimesh.load("face_mesh_poisson_me.ply")

# Export to GLB
mesh.export("face_mesh_poisson_me.glb")
