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

########

# import cv2
# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt

# depth_map_path = "./MODNet/depth_map/me2_depth_map.png"
# hair_mask_path = "./face-parsing.PyTorch/hair_masks/me2_hair_mask.png"
# original_image_path = "./images/me2.jpg"
# output_mesh_path = "./obj_files/me2_colored_hair_mesh.obj"

# fx, fy = 500, 500
# cx, cy = 128, 128  # For 256x256

# # === Load Inputs ===
# depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

# hair_mask = cv2.imread(hair_mask_path, cv2.IMREAD_GRAYSCALE)
# original_img = cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB)

# # Resize to match
# H, W = depth_map.shape
# hair_mask = cv2.resize(hair_mask, (W, H))
# original_img = cv2.resize(original_img, (W, H))

# # === Normalize and Expand Depth ===
# depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
# depth_map = depth_map ** 0.8  # Slight exaggeration of depth
# depth_map *= 120  # Increase max depth (was 100)

# # === Refine Mask ===
# _, refined_mask = cv2.threshold(hair_mask, 80, 255, cv2.THRESH_BINARY)
# refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
# refined_mask = cv2.dilate(refined_mask, np.ones((5, 5), np.uint8), iterations=2)

# # === Generate 3D Points ===
# points, colors = [], []
# for y in range(H):
#     for x in range(W):
#         if refined_mask[y, x] > 0:
#             Z = depth_map[y, x]
#             if Z <= 0.01: continue
#             X = (x - cx) * Z / fx
#             Y = (y - cy) * Z / fy
#             points.append([X, -Y, -Z])  # Orientation fix
#             colors.append(original_img[y, x] / 255.0)

# # === Create Point Cloud ===
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)
# pcd.estimate_normals()

# # Optional: View raw point cloud
# o3d.visualization.draw_geometries([pcd], window_name="Hair Point Cloud")

# # === Build Mesh ===
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radii = [avg_dist * x for x in [1.0, 1.5, 2.0]]

# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd,
#     o3d.utility.DoubleVector(radii)
# )

# # === Smooth and Save ===
# mesh.compute_vertex_normals()
# mesh = mesh.filter_smooth_simple(number_of_iterations=5)
# mesh.vertex_colors = pcd.colors

# o3d.io.write_triangle_mesh(output_mesh_path, mesh)
# print(f"[✔] Colored hair mesh saved: {output_mesh_path}")

# o3d.visualization.draw_geometries([mesh], window_name="Smoothed Colored Hair Mesh", mesh_show_back_face=True)

 #######
'''import numpy as np
import open3d as o3d
from PIL import Image
import os
from tqdm import tqdm
import cv2

def reconstruct_3d_hair_from_depth_and_mask(rgb_image_path, depth_map_path, hair_mask_path, output_obj_folder, camera_intrinsics=None):
    try:
        rgb_img = Image.open(rgb_image_path).convert("RGB")
        depth_map_raw = Image.open(depth_map_path).convert("L") # 'L' for grayscale
        hair_mask_raw = Image.open(hair_mask_path).convert("L") # 'L' for grayscale
        print("DEBUG: Images loaded successfully.")
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load or process input images: {e}")
        return

    depth_data = np.array(depth_map_raw).astype(np.float32) # Values 0-255 (assuming this range for depth map PNG)
    hair_mask_data = np.array(hair_mask_raw).astype(np.float32) / 255.0 # Normalize mask to 0-1

    H, W = depth_data.shape
    print(f"DEBUG: Image dimensions (H, W): ({H}, {W})")

    print(f"DEBUG: Hair Mask Data (min, max): ({np.min(hair_mask_data)}, {np.max(hair_mask_data)})")
    print(f"DEBUG: Depth Data (raw, min, max): ({np.min(depth_data)}, {np.max(depth_data)})")
    
    if np.sum(hair_mask_data > 0.5) < 10: # Very few hair pixels
        print("WARNING: Very few hair pixels found in mask. Output might be empty or distorted.")
        
    min_depth_val_in_meters = 0.5  
    max_depth_val_in_meters = 1.5  

    depth_data = cv2.bilateralFilter(depth_data.astype(np.uint8), d=5, sigmaColor=75, sigmaSpace=75)
    depth_data = depth_data.astype(np.float32)

    depth_normalized_pixel_value = depth_data / 255.0
    depth_in_meters = min_depth_val_in_meters + (1.0 - depth_normalized_pixel_value) * (max_depth_val_in_meters - min_depth_val_in_meters)

    print(f"DEBUG: Depth in Meters (min, max): ({np.min(depth_in_meters)}, {np.max(depth_in_meters)})")

    if camera_intrinsics is None:
        focal_length = W * 1.0 # Adjust this. Try W * 0.8, W * 1.2, W * 1.5
        cx, cy = W / 2, H / 2
        
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H, fx=focal_length, fy=focal_length, cx=cx, cy=cy
        )
        print(f"DEBUG: Using estimated camera intrinsics: fx={focal_length}, fy={focal_length}, cx={cx}, cy={cy}")
    else:
        intrinsic_matrix = camera_intrinsics
        print(f"DEBUG: Using provided camera intrinsics: {intrinsic_matrix}")


    # --- 4. Create 3D Point Cloud ---
    points = []
    colors = []
    skipped_points_count = 0 
    
    # Check if there are any hair pixels in the mask before looping
    if np.sum(hair_mask_data > 0.5) == 0:
        print("WARNING: Hair mask is completely empty. No points to reconstruct.")
        return None # Return None if no hair found

    for v in tqdm(range(H), desc="Generating Hair Point Cloud"):
        for u in range(W):
            if hair_mask_data[v, u] > 0.5: # If pixel is part of the hair mask
                z = depth_in_meters[v, u] # Get depth value (Z-coordinate)

                if not np.isfinite(z) or z <= 0 or z > max_depth_val_in_meters * 2: 
                    skipped_points_count += 1 
                    continue 
                x = (u - intrinsic_matrix.get_principal_point()[0]) * z / intrinsic_matrix.get_focal_length()[0]
                y = (v - intrinsic_matrix.get_principal_point()[1]) * z / intrinsic_matrix.get_focal_length()[1]

                points.append([x, y, z])
                colors.append(np.array(rgb_img.getpixel((u, v))) / 255.0) # Normalize RGB to 0-1
    print(f"DEBUG: Generated hair point cloud with {len(points)} points. SKIPPED {skipped_points_count} points due to invalid Z-values.") # <-- ENSURE THIS IS ACTIVE
    if not points:
        print("ERROR: No valid 3D points could be generated for hair. Check mask, depth, or intrinsics.")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd = pcd.voxel_down_sample(voxel_size=0.0025)
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    print(f"DEBUG: Generated hair point cloud with {len(points)} points. SKIPPED {skipped_points_count} points due to invalid Z-values.") 

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30) 
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    print("Normals estimated and oriented.")
    # o3d.visualization.draw_geometries([pcd], window_name="Hair Point Cloud - Inspect Quality"
    print("Generating hair mesh using Ball Pivoting Reconstruction...")

    radii = [0.005, 0.01, 0.02]  
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        print("  Mesh generated using Ball Pivoting.")
    except Exception as e:
        print(f"ERROR: Ball Pivoting failed: {e}")
        return None

    output_filename = os.path.splitext(os.path.basename(rgb_image_path))[0] + "_hair.obj"
    output_full_path = os.path.join(output_obj_folder, output_filename)

    try:
        o3d.io.write_triangle_mesh(output_full_path, mesh)
        print(f"3D hair mesh saved to: {output_full_path}")
    except Exception as e:
        print(f"ERROR: Failed to save mesh to {output_full_path}: {e}")
        return None
    o3d.visualization.draw_geometries([mesh], window_name=f"Reconstructed Hair: {output_filename}")

    print("\nReconstruction process finished.")
    return mesh

if __name__ == "__main__":

    my_rgb_image_path = "./images/fimg8.jpg" 
    my_depth_map_path = "./MODNet/depth_map/fimg8_depth_map.png" 
    my_hair_mask_path = "./face-parsing.PyTorch/hair_masks/fimg8_hair_mask.png" 
    my_output_folder = "./obj_files/" 

    os.makedirs(my_output_folder, exist_ok=True) 

    # --- Limit OMP threads for stability on macOS (Keep this) ---
    os.environ["OMP_NUM_THREADS"] = "1" 

    reconstruct_3d_hair_from_depth_and_mask(
        my_rgb_image_path,
        my_depth_map_path,
        my_hair_mask_path,
        my_output_folder
    )
    print("\nScript finished generating 3D hair mesh.")

    # --- You can remove or comment out the argparse section ---
    # parser = argparse.ArgumentParser(description="Generate 3D hair mesh from RGB, Depth, and Mask.")
    # parser.add_argument("--rgb_path", type=str, required=True, ...)
    # parser.add_argument("--depth_path", type=str, required=True, ...)
    # parser.add_argument("--mask_path", type=str, required=True, ...)
    # parser.add_argument("--output_folder", type=str, default="./output_3d_hair_meshes/", ...)
    # args = parser.parse_args()
    # (The `args` variable would no longer be used if you remove the parser section) '''

import numpy as np
import open3d as o3d
from PIL import Image
import os
from tqdm import tqdm
import cv2

def reconstruct_3d_hair_from_depth_and_mask(rgb_image_path, depth_map_path, hair_mask_path, output_obj_folder, camera_intrinsics=None):
    # --- 1. Load images ---
    rgb_img = Image.open(rgb_image_path).convert("RGB")
    depth_map_raw = Image.open(depth_map_path).convert("L")
    hair_mask_raw = Image.open(hair_mask_path).convert("L")

    depth_data = np.array(depth_map_raw).astype(np.float32)
    hair_mask_data = np.array(hair_mask_raw).astype(np.float32) / 255.0

    H, W = depth_data.shape
    print(f"DEBUG: Image dimensions (H, W): ({H}, {W})")
    print(f"DEBUG: Hair Mask Data (min, max): ({np.min(hair_mask_data)}, {np.max(hair_mask_data)})")
    print(f"DEBUG: Depth Data (raw, min, max): ({np.min(depth_data)}, {np.max(depth_data)})")

    if np.sum(hair_mask_data > 0.5) < 10:
        print("WARNING: Very few hair pixels found in mask.")

    min_depth_val_in_meters = 0.5
    max_depth_val_in_meters = 1.5

    depth_data = cv2.bilateralFilter(depth_data.astype(np.uint8), d=5, sigmaColor=75, sigmaSpace=75)
    depth_data = depth_data.astype(np.float32)

    depth_normalized_pixel_value = depth_data / 255.0
    depth_in_meters = min_depth_val_in_meters + (1.0 - depth_normalized_pixel_value) * (max_depth_val_in_meters - min_depth_val_in_meters)

    print(f"DEBUG: Depth in Meters (min, max): ({np.min(depth_in_meters)}, {np.max(depth_in_meters)})")

    # --- 2. Set up camera intrinsics ---
    if camera_intrinsics is None:
        focal_length = W * 1.0
        cx, cy = W / 2, H / 2
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=focal_length, fy=focal_length, cx=cx, cy=cy)
    else:
        intrinsic_matrix = camera_intrinsics

    # --- 3. Generate point cloud ---
    points = []
    colors = []
    for v in tqdm(range(H), desc="Generating Hair Point Cloud"):
        for u in range(W):
            if hair_mask_data[v, u] > 0.5:
                z = depth_in_meters[v, u]
                if not np.isfinite(z) or z <= 0 or z > max_depth_val_in_meters * 2:
                    continue
                x = (u - intrinsic_matrix.get_principal_point()[0]) * z / intrinsic_matrix.get_focal_length()[0]
                y = (v - intrinsic_matrix.get_principal_point()[1]) * z / intrinsic_matrix.get_focal_length()[1]
                points.append([x, y, z])
                colors.append(np.array(rgb_img.getpixel((u, v))) / 255.0)

    if not points:
        print("ERROR: No valid 3D points generated.")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd = pcd.voxel_down_sample(voxel_size=0.0025)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    print("Generating mesh using Ball Pivoting...")
    radii = [0.005, 0.01, 0.02]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # 🔧 Step 1: Crop by Z (remove backside)
    bbox = mesh.get_axis_aligned_bounding_box()
    z_max = bbox.get_max_bound()[2]
    z_threshold = z_max - 0.1  # keep only front portion
    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(bbox.get_min_bound()[0], bbox.get_min_bound()[1], bbox.get_min_bound()[2]),
        max_bound=(bbox.get_max_bound()[0], bbox.get_max_bound()[1], z_threshold)
    )
    mesh = mesh.crop(crop_box)
    print("🔧 Backside removed using Z cropping.")

    # 🔧 Step 2: Keep only the largest cluster
    # Cluster connected triangles
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    largest_cluster_idx = np.argmax(cluster_n_triangles)

    # Get indices of triangles to keep
    triangles_to_keep = np.where(triangle_clusters == largest_cluster_idx)[0]

    # Create a new mesh using only the largest triangle cluster
    mesh_filtered = o3d.geometry.TriangleMesh()
    mesh_filtered.vertices = mesh.vertices
    mesh_filtered.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[triangles_to_keep])
    mesh_filtered.vertex_colors = mesh.vertex_colors
    mesh_filtered.compute_vertex_normals()

    mesh = mesh_filtered  # replace original mesh with the filtered one

    print("🔧 Kept only the largest connected mesh component.")

    # 🔧 Step 3: Smooth the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()
    print("🔧 Mesh smoothed.")

    output_filename = os.path.splitext(os.path.basename(rgb_image_path))[0] + "_hair.obj"
    output_full_path = os.path.join(output_obj_folder, output_filename)
    o3d.io.write_triangle_mesh(output_full_path, mesh)
    print(f"3D hair mesh saved to: {output_full_path}")

    o3d.visualization.draw_geometries([mesh], window_name=f"Reconstructed Hair: {output_filename}")
    return mesh

if __name__ == "__main__":
    my_rgb_image_path = "./images/dme.jpg"
    my_depth_map_path = "./MODNet/depth_map/dme_depth_map.png"
    my_hair_mask_path = "./face-parsing.PyTorch/hair_masks/dme_hair_mask.png"
    my_output_folder = "./obj_files/"
    os.makedirs(my_output_folder, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    reconstruct_3d_hair_from_depth_and_mask(
        my_rgb_image_path,
        my_depth_map_path,
        my_hair_mask_path,
        my_output_folder
    )
    print("\n Script finished generating 3D hair mesh.")
