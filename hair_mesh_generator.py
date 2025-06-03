import cv2
import numpy as np
import open3d as o3d

# === 1. Load depth map and hair mask ===
depth_map_path = "./hair_depth_masks/depth_map/me2_depth_map.png"
hair_mask_path = "./hair_depth_masks/hair_masks/me2_hair_mask.png"

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
o3d.io.write_triangle_mesh("./ply_files/me2_hair_mesh.ply", mesh_crop)
print("Saved mesh as hair_mesh.ply")
