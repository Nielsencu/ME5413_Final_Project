import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Your point cloud loading
pcd = o3d.io.read_point_cloud("points/fastlio2.pcd")
color_bar = "tab20c"

# Apply new colors (assuming based on z-coordinate as an example)
z = np.asarray(pcd.points)[:, 2]
normalized_z = (z - np.min(z)) / (np.max(z) - np.min(z))
new_colors = plt.get_cmap(color_bar)(normalized_z)[:, :3]  # New colors from colormap

# Create a new point cloud object with the new colors
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = pcd.points  # Reuse the original points
new_pcd.colors = o3d.utility.Vector3dVector(new_colors)  # Apply the new colors

# Proceed with visualization of new_pcd instead of pcd
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(new_pcd)
opts = vis.get_render_option()
opts.point_size = 3

# avaliable opts are:
opts.background_color = np.asarray([0, 0, 0])

vis.run()
vis.destroy_window()
