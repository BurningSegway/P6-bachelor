import numpy as np
import laspy as lp
import open3d as o3d

point_cloud = lp.read("2020_Drone_M.las")

print([dimension.name for dimension in point_cloud.point_format.dimensions])
print(np.max(point_cloud.red))

points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

factor = 8
decimated_points_random = points[::factor]
decimated_colors_random = colors[::factor]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(decimated_points_random)
pcd.colors = o3d.utility.Vector3dVector(decimated_colors_random/65535)

o3d.visualization.draw_geometries([pcd])
