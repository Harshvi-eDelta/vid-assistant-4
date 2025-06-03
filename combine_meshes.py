import open3d as o3d

# === PATHS ===
face_mesh_path = "./FaceVerse_v4/ply_files/face_mesh_poisson_fimg4.ply"     # PLY format face mesh
hair_mesh_path = "./obj_files/fimg4_colored_hair_mesh.obj"     # OBJ format hair mesh
output_combined_path = "./final_combine_face/fimg4_combined_face_hair_mesh.obj"

# === LOAD MESHES ===
face_mesh = o3d.io.read_triangle_mesh(face_mesh_path)
hair_mesh = o3d.io.read_triangle_mesh(hair_mesh_path)

# === OPTIONAL: Normalize if needed
face_mesh.compute_vertex_normals()
hair_mesh.compute_vertex_normals()

# === COMBINE ===
combined_mesh = face_mesh + hair_mesh

# === SAVE COMBINED MESH ===
o3d.io.write_triangle_mesh(output_combined_path, combined_mesh)
print(f"Combined mesh saved to {output_combined_path}")

# === VISUALIZE ===
o3d.visualization.draw_geometries([combined_mesh], mesh_show_back_face=True)
