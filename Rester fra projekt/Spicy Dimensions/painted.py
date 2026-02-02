import open3d as o3d
import numpy as np
import matplotlib as plt

pcd = o3d.io.read_point_cloud("/home/pierre/Desktop/Spicy Dimensions/cropped_1.ply")

# Visualize the point cloud

pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
radius_normals = nn_distance*4

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn = 16), fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])

o3d.visualization.draw_geometries([pcd])
