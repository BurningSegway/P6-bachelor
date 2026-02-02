import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("/home/pierre/Desktop/Spicy Dimensions/cropped_1.ply")

# Visualize the point cloud

pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())
radius_normals = nn_distance*4

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn = 16), fast_normal_computation=True)
pcd.paint_uniform_color([0.6, 0.6, 0.6])

pt_to_plane_dist = 0.002

plane_model, inliers = pcd.segment_plane(distance_threshold=pt_to_plane_dist, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert = True)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        outlier_cloud.cluster_dbscan(eps=0.005, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([outlier_cloud])
