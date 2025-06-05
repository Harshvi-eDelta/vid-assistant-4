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



import numpy as np
import open3d as o3d
from PIL import Image
import os
from tqdm import tqdm

import numpy as np
import open3d as o3d
from PIL import Image
import os
from tqdm import tqdm # Assuming tqdm is now correctly imported

def reconstruct_3d_hair_from_depth_and_mask(rgb_image_path, depth_map_path, hair_mask_path, output_obj_path, camera_intrinsics=None):
    print("Starting 3D hair reconstruction (programmatic, CPU-only)...")
    print(f"Input RGB: {rgb_image_path}")
    print(f"Input Depth Map: {depth_map_path}")
    print(f"Input Hair Mask: {hair_mask_path}")
    print(f"Output OBJ Path: {output_obj_path}")

    # --- 1. Load Images ---
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

    depth_data = np.array(depth_map_raw).astype(np.float32) # Values 0-255
    hair_mask_data = np.array(hair_mask_raw).astype(np.float32) / 255.0 # Normalize to 0-1

    H, W = depth_data.shape
    print(f"DEBUG: Image dimensions (H, W): ({H}, {W})")

    # --- Debug: Check mask and depth data ranges ---
    print(f"DEBUG: Hair Mask Data (min, max): ({np.min(hair_mask_data)}, {np.max(hair_mask_data)})")
    print(f"DEBUG: Depth Data (raw, min, max): ({np.min(depth_data)}, {np.max(depth_data)})")
    
    # Check if mask is essentially empty
    if np.sum(hair_mask_data > 0.5) < 10: # Very few hair pixels
        print("WARNING: Very few hair pixels found in mask. Output might be empty or distorted.")
        
    # --- 2. Normalize Depth Data (CRITICAL SECTION) ---
    # Adjust these values based on your scene.
    # For your depth map (white is closer, black is further):
    # - 255 (white) should map to min_depth_val_in_meters
    # - 0 (black) should map to max_depth_val_in_meters
    
    min_depth_val_in_meters = 0.5  # Assumed closest point in real-world meters
    max_depth_val_in_meters = 1.5  # Assumed furthest point in real-world meters

    # Map 0-255 to 0-1 (normalized pixel value)
    depth_normalized_pixel_value = depth_data / 255.0

    # Map normalized pixel value to depth in meters.
    # Since higher pixel value (white) is closer, use (1.0 - pixel_value) * range + min_val
    depth_in_meters = min_depth_val_in_meters + (1.0 - depth_normalized_pixel_value) * (max_depth_val_in_meters - min_depth_val_in_meters)

    print(f"DEBUG: Depth in Meters (min, max): ({np.min(depth_in_meters)}, {np.max(depth_in_meters)})")

    # --- 3. Define Camera Intrinsics ---
    if camera_intrinsics is None:
        # These are rough assumptions. For accurate results, you need calibration data.
        # A common approximation for focal length is 1.2 * max(W, H) or W / (2 * tan(FOV/2))
        # Given your 512x512 images, focal_length ~500-600 pixels is common.
        focal_length = W * 1.0 # Or try W / 1.5 or W * 1.2
        cx, cy = W / 2, H / 2
        
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
            width=W, height=H, fx=focal_length, fy=focal_length, cx=cx, cy=cy
        )
        print(f"DEBUG: Using estimated camera intrinsics: fx={focal_length}, fy={focal_length}, cx={cx}, cy={cy}")
    else:
        intrinsic_matrix = camera_intrinsics
        print(f"DEBUG: Using provided camera intrinsics: {intrinsic_matrix}")


    # --- 4. Create Point Cloud ---
    points = []
    colors = []
    
    # Check if there are any hair pixels in the mask before looping
    if np.sum(hair_mask_data > 0.5) == 0:
        print("WARNING: Hair mask is completely empty. No points to reconstruct.")
        return # Exit if no hair found

    for v in tqdm(range(H), desc="Generating Point Cloud"):
        for u in range(W):
            if hair_mask_data[v, u] > 0.5: # If pixel is part of the hair mask
                z = depth_in_meters[v, u] # Get depth value (Z-coordinate)

                # Ensure z is a valid number (not NaN, Inf, or extremely small/large)
                if not np.isfinite(z) or z <= 0 or z > max_depth_val_in_meters * 2: # Check for invalid depth
                    continue # Skip invalid points

                # Convert 2D pixel (u,v) and depth (z) to 3D point (x,y,z)
                x = (u - intrinsic_matrix.get_principal_point()[0]) * z / intrinsic_matrix.get_focal_length()[0]
                y = (v - intrinsic_matrix.get_principal_point()[1]) * z / intrinsic_matrix.get_focal_length()[1]

                points.append([x, y, z])
                colors.append(np.array(rgb_img.getpixel((u, v))) / 255.0) # Normalize RGB to 0-1

    if not points:
        print("ERROR: No valid 3D points could be generated. This might be due to an empty mask, invalid depth values, or incorrect intrinsics.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.visualization.draw_geometries([pcd])
    print(f"DEBUG: Generated point cloud with {len(points)} points.")

    # --- 5. (Optional) Downsample and estimate normals for meshing ---
    # Downsampling can speed up meshing, but might lose detail
    # pcd = pcd.voxel_down_sample(voxel_size=0.01) # Adjust voxel_size based on your scale

    # Estimate normals for surface reconstruction
    # Adjust radius/max_nn based on point density and desired smoothness
    # Use a radius that's appropriate for the scale of your points (e.g., 0.05-0.1 for points in meter range)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    print("DEBUG: Normals estimated for point cloud.")

    # --- 6. Surface Reconstruction (Generate Mesh) ---
    # Poisson reconstruction parameters. Adjust `depth` (higher is more detail, more noise).
    # `depth=9` is already quite high for typical point clouds.
    # If point cloud is noisy, lower `depth` to smooth it out (e.g., 6-8).
    # If the point cloud is sparse, create_from_point_cloud_poisson can fail.
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6) # Changed depth to 8 for robustness
        print("DEBUG: Mesh generated using Poisson Reconstruction.")

        # You might want to filter the mesh by density to remove spurious geometry
        # vertices_to_remove = densities < np.quantile(densities, 0.01) # Remove bottom 1% by density
        # mesh.remove_vertices_by_mask(vertices_to_remove)

    except Exception as e:
        print(f"ERROR: Mesh generation failed using Poisson Reconstruction: {e}")
        print("This often happens if the point cloud is very noisy, sparse, or has incorrect scale.")
        return


    # --- 7. Save the Mesh ---
    output_obj_file = os.path.join(output_obj_path, os.path.splitext(os.path.basename(rgb_image_path))[0] + "_hair.obj")
    try:
        o3d.io.write_triangle_mesh(output_obj_file, mesh)
        print(f"3D hair mesh saved to: {output_obj_file}")
    except Exception as e:
        print(f"ERROR: Failed to save mesh: {e}")
        return

    # Optional: Visualize the mesh for debugging
    o3d.visualization.draw_geometries([mesh])

    print("\nReconstruction process finished.")


# --- Example Usage (replace with your actual paths) ---
if __name__ == "__main__":
    # Make sure your files exist
    # Put your original RGB image here
    rgb_img_path = "./images/dme.jpg"
    # Put your MODNet depth map here
    depth_map_path = "./MODNet/depth_map/dme_depth_map.png"
    # Put your face-parsing hair mask here
    hair_mask_path = "./face-parsing.PyTorch/hair_masks/dme_hair_mask.png"
    
    # Output directory for the OBJ file
    output_folder = "./obj_files/dme_colored_hair_mesh.obj"
    os.makedirs(output_folder, exist_ok=True)

    reconstruct_3d_hair_from_depth_and_mask(
        rgb_img_path,
        depth_map_path,
        hair_mask_path,
        output_folder
    )
    print("\nReconstruction process finished.")
