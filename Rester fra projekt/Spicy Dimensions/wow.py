import open3d as o3d
import numpy as np
import cv2

# Load the PNG depth image
depth_image_path = "/home/pierre/Desktop/Spicy Dimensions/Image_depth_2023_12_13_11_36_1.png"
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Load as is (e.g., 16-bit depth)

# Check depth image properties
print(f"Depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")

# Convert to meters if necessary (assuming depth is in millimeters)
depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters

# Create an Open3D Image
o3d_depth_image = o3d.geometry.Image(depth_image)

# Create an Open3D point cloud

# Define approximate intrinsic parameters
width = depth_image.shape[1]  # e.g., 640
height = depth_image.shape[0]  # e.g., 480
fx = 525.0  # Approximate focal length in pixels
fy = 525.0  # Approximate focal length in pixels
cx = width / 2.0  # Principal point x-coordinate
cy = height / 2.0  # Principal point y-coordinate

camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=width,
    height=height,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
)


# Define the intrinsic parameters of your camera
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    width=depth_image.shape[1],
    height=depth_image.shape[0],
    fx=fx,  # Replace with your camera's focal length in x
    fy=fy,  # Replace with your camera's focal length in y
    cx=cx,  # Replace with your camera's principal point x
    cy=cy,  # Replace with your camera's principal point y
)

# Generate the point cloud from the depth image
pcd = o3d.geometry.PointCloud.create_from_depth_image(
    depth=o3d_depth_image,
    intrinsic=camera_intrinsics,
    extrinsic=np.eye(4),  # Use an identity matrix if no transformation is needed
)

#diameter = np.linalg.norm(
#    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

# Visualize the point cloud
#o3d.visualization.draw_geometries_with_editing([pcd])
vis = o3d.visualization.VisualizerWithEditing(-1, False, "")
vis.create_window()
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

#camera = [0, 0, diameter]
#radius = diameter * 100

#print("Get all points that are visible from given view point")
#_, pt_map = pcd.hidden_point_removal(camera, radius)

#print("Visualize result")
#pcd = pcd.select_by_index(pt_map)
#o3d.visualization.draw_geometries([pcd],
#                                  zoom=0.19999999999999959,
#                                  front=[ 0.36890262514852351, -0.7362016774863912, -0.56737813072478549 ],
#                                  lookat=[ -0.010628314150525847, 0.075031977817612408, 0.59350557497726564 ],
#                                  up=[ -0.25076478982532374, 0.50896449155810253, -0.82345137471308139 ])

# Save the point cloud to a file
#o3d.io.write_point_cloud("output_point_cloud.ply", point_cloud)