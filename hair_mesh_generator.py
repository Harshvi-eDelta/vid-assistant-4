'''import cv2
import numpy as np
import open3d as o3d

# === 1. Load depth map and hair mask ===
depth_map_path = "./hair_depth_masks/hair_masks/depth_map/sh_depth_map.png"
hair_mask_path = "./hair_depth_masks/hair_masks/sh_hair_mask.png"

depth = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)

# Ensure both are same size
assert depth.shape == mask.shape, "Depth map and mask must be the same size."

# Normalize depth map
depth = depth.astype(np.float32) / 255.0

# Resize or crop if needed (optional)
H, W = depth.shape

# === 2. Mask the depth map to only keep hair region ===
masked_depth = np.where(mask > 128, depth, 0.0)

# === 3. Generate 3D point cloud ===
points = []
colors = []

fx, fy = 1.0, 1.0  # Dummy intrinsics (can be refined if real camera used)
cx, cy = W / 2, H / 2

for y in range(H):
    for x in range(W):
        d = masked_depth[y, x]
        if d > 0:
            # Backproject (x, y, d) to 3D
            X = (x - cx) * d / fx
            Y = (y - cy) * d / fy
            Z = d
            points.append([X, -Y, -Z])  # Flip Y/Z for correct view

points = np.array(points)

# === 4. Create Open3D point cloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optional: estimate normals
pcd.estimate_normals()

# === 5. Surface Reconstruction (Poisson) ===
print("Running Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
mesh.compute_vertex_normals()

# === 6. Crop mesh to remove artifacts ===
bbox = pcd.get_axis_aligned_bounding_box()
mesh_crop = mesh.crop(bbox)

# === 7. Save mesh ===
o3d.io.write_triangle_mesh("./ply_files/sh_hair_mesh.ply", mesh_crop)
print("Saved mesh ")

# Visualize
o3d.visualization.draw_geometries([mesh_crop]) '''

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

depth_map_path = "./MODNet/depth_map/me2_depth_map.png"
hair_mask_path = "./face-parsing.PyTorch/hair_masks/me2_hair_mask.png"
original_image_path = "./images/me2.jpg"
output_mesh_path = "./obj_files/me2_colored_hair_mesh.obj"

fx, fy = 500, 500
cx, cy = 128, 128  # For 256x256

# === Load Inputs ===
depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
original_img = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)

# Resize to match
H, W = depth_map.shape
hair_mask = cv2.resize(hair_mask, (W, H))
original_img = cv2.resize(original_img, (W, H))

# === Normalize and Expand Depth ===
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_map = depth_map ** 0.8  # Slight exaggeration of depth
depth_map *= 120  # Increase max depth (was 100)

# === Refine Mask ===
_, refined_mask = cv2.threshold(hair_mask, 80, 255, cv2.THRESH_BINARY)
refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
refined_mask = cv2.dilate(refined_mask, np.ones((5, 5), np.uint8), iterations=2)

# === Generate 3D Points ===
points, colors = [], []
for y in range(H):
    for x in range(W):
        if refined_mask[y, x] > 0:
            Z = depth_map[y, x]
            if Z <= 0.01: continue
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy
            points.append([X, -Y, -Z])  # Orientation fix
            colors.append(original_img[y, x] / 255.0)

# === Create Point Cloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.estimate_normals()

# Optional: View raw point cloud
o3d.visualization.draw_geometries([pcd], window_name="Hair Point Cloud")

# === Build Mesh ===
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radii = [avg_dist * x for x in [1.0, 1.5, 2.0]]

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd,
    o3d.utility.DoubleVector(radii)
)

# === Smooth and Save ===
mesh.compute_vertex_normals()
mesh = mesh.filter_smooth_simple(number_of_iterations=5)
mesh.vertex_colors = pcd.colors

o3d.io.write_triangle_mesh(output_mesh_path, mesh)
print(f"[✔] Colored hair mesh saved: {output_mesh_path}")

o3d.visualization.draw_geometries([mesh], window_name="Smoothed Colored Hair Mesh", mesh_show_back_face=True)



