import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/home/pierre/Desktop/Spicy Dimensions/cropped_1.ply")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd],
                                               zoom=0.25999999999999956,
                                               front=[ 0.26913707633851214, 0.60545883934446054, -0.74898920419430404 ],
                                               lookat=[ -0.010334136061942192, 0.045325207287089088, 0.613643775916344 ],
                                               up=[ -0.25701040982469409, -0.7043317785919726, -0.66171171586062416 ])


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
