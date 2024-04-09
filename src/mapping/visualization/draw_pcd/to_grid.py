import open3d as o3d
import numpy as np

# Parameters
grid_size = 1000
head_level = 0.1  # Adjustable head level (z-coordinate)

# Read the point cloud
pcd = o3d.io.read_point_cloud("points/FAST-LIO.pcd")

# convert every point from -z, x, y to x, y, z
# points = np.asarray(pcd.points)
# points[:, [0, 1, 2]] = points[:, [0, 2, 1]]
# pcd.points = o3d.utility.Vector3dVector(points)

# Get the points
points = np.asarray(pcd.points)

# Assuming the point cloud is already aligned with the world axes and
# centered, otherwise you might need to transform or filter it.

# Create a grid
grid = np.zeros((grid_size, grid_size), dtype=np.int32)

# Define the grid boundaries (you might want to adjust these based on your point cloud)
x_min, y_min, x_max, y_max = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()

# Function to convert point position to grid coordinates
def point_to_grid(x, y, x_min, y_min, x_max, y_max, grid_size):
    # Scale and transform the point coordinates to grid indices
    grid_x = int(grid_size * (x - x_min) / (x_max - x_min))
    grid_y = int(grid_size * (y - y_min) / (y_max - y_min))
    return grid_x, grid_y

# Populate the grid
for point in points:
    if point[2] > head_level:
        x, y = point_to_grid(point[0], point[1], x_min, y_min, x_max, y_max, grid_size)
        if 0 <= x < grid_size and 0 <= y < grid_size:

            var = 5
            intensity = 50
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if 0 <= x+i < grid_size and 0 <= y+j < grid_size:
                        grid[x+i, y+j] += intensity * np.exp(- (i**2 + j**2) / (2 * var**2))

# Normalize the grid values to the range 0-255 for visualization
grid_normalized = (grid / grid.max()) * 255 * 10
grid_normalized[grid_normalized > 255] = 255
grid_normalized = grid_normalized.astype(np.uint8)

# dilate
# from scipy.ndimage import binary_dilation
# grid_normalized = binary_dilation(grid_normalized, iterations=1)

# invert
grid_normalized = 255 - grid_normalized

# Save the grid as an image
from PIL import Image
image = Image.fromarray(grid_normalized)
image.save('point_cloud_grid.png')
