import open3d as o3d
import numpy as np
import yaml

# Parameters
grid_size = 1000
head_level = 0.1  # Adjustable head level (z-coordinate)
# resolution = 0.1  # Resolution of the map in meters per pixel

# Read the point cloud
name = "SC-DLO"
pcd = o3d.io.read_point_cloud(f"points/{name}.pcd")

# rotate -90 degrees in xy plane, i.e. x -> y, y -> -x
points = np.asarray(pcd.points)
points[:, [0, 1]] = points[:, [1, 0]]
points[:, 0] = -points[:, 0]
pcd.points = o3d.utility.Vector3dVector(points)

# Get the points
points = np.asarray(pcd.points)

# Create a grid
grid = np.zeros((grid_size, grid_size), dtype=np.int32)

# Define the grid boundaries
x_min, y_min, x_max, y_max = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()

resolution = (x_max - x_min) / grid_size
print(f"Resolution: {resolution}")

# Convert point position to grid coordinates
def point_to_grid(x, y, x_min, y_min, x_max, y_max, grid_size):
    grid_x = int(grid_size * (x - x_min) / (x_max - x_min))
    grid_y = int(grid_size * (y - y_min) / (y_max - y_min))
    return grid_x, grid_y

# Populate the grid
for point in points:
    if point[2] > head_level:
        x, y = point_to_grid(point[0], point[1], x_min, y_min, x_max, y_max, grid_size)
        if 0 <= x < grid_size and 0 <= y < grid_size:
            # grid[x, y] += 1
            # for i in range(-1, 2):
            #     for j in range(-1, 2):
            #         grid[x+i, y+j] += 0.5

            # draw a gaussian
            var = 0.5
            intensity = 1
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if 0 <= x+i < grid_size and 0 <= y+j < grid_size:
                        grid[x+i, y+j] += int(intensity * np.exp(- (i**2 + j**2) / (2 * var**2)))

# Normalize the grid values to the range 0-255 for visualization
grid_normalized = (grid / grid.max()) * 255 * 10000
grid_normalized[grid_normalized > 255] = 255
grid_normalized = grid_normalized.astype(np.uint8)

# Invert the grid to follow ROS conventions for occupancy grids
grid_normalized = 255 - grid_normalized

# erode 3x3
import cv2
kernel = np.ones((3, 3), np.uint8)
grid_normalized = cv2.erode(grid_normalized, kernel, iterations=4)
grid_normalized = cv2.dilate(grid_normalized, kernel, iterations=4)

# Save the grid as a PGM file
pgm_path = f'./{name}.pgm'
with open(pgm_path, 'wb') as pgm_file:
    pgm_file.write(b'P5\n# CREATOR: Open3D and NumPy\n')
    pgm_file.write(f'{grid_size} {grid_size}\n255\n'.encode())
    grid_normalized.tofile(pgm_file)

# Create and save YAML file
yaml_path = f'{name}.yaml'
map_metadata = {
    'image': pgm_path,
    'resolution': resolution,
    'origin': [x_min, y_min, 0],  # Adjust as necessary
    'occupied_thresh': 0.588,  # Adjust as necessary
    'free_thresh': 0.196,  # Adjust as necessary
    'negate': 0
}

# show a histogram of the grid values
# import matplotlib.pyplot as plt
# plt.hist(grid_normalized.flatten(), bins=50)
# plt.show()

with open(yaml_path, 'w') as yaml_file:
    for key, value in map_metadata.items():
        yaml_file.write(f'{key}: {value}\n')