import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Your point cloud loading
name = "DLO"
pcd = o3d.io.read_point_cloud(f"points/{name}.pcd")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
bbox = pcd.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)

# uniform_color = [1, 0, 0]  # Red
# p_mesh_crop.vertex_colors = o3d.utility.Vector3dVector([uniform_color] * len(p_mesh_crop.vertices))

# Get the z-coordinates of the vertices
z = np.asarray(p_mesh_crop.vertices)[:, 2]
normalized_z = (z - np.min(z)) / (np.max(z) - np.min(z))

# Apply colors based on z-coordinate using a colormap
color_map = plt.get_cmap('tab20c')
mesh_colors = color_map(normalized_z)[:, :3]  # Get RGB values from colormap

# Assign the colors to the mesh
p_mesh_crop.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)


vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(p_mesh_crop)
opts = vis.get_render_option()
opts.point_size = 10



vis.run()
vis.destroy_window()

# save mesh as obj
# o3d.io.write_triangle_mesh(f"meshes/{name}.obj", p_mesh_crop)

# save mesh as glb
o3d.io