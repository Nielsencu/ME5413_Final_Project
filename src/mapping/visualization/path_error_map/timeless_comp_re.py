import os
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def read_tum_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 8:
                x, y, z = map(float, parts[1:4])
                data.append((x, y, z))
    return data

def calculate_distances(data):
    distances = [0.0]
    for i in range(1, len(data)):
        prev_point = data[i - 1]
        curr_point = data[i]
        distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2 + (curr_point[2] - prev_point[2])**2)
        distances.append(distances[-1] + distance)
    return distances

def resample_path_uniformly(data, step):
    distances = calculate_distances(data)
    max_distance = distances[-1]

    # Create new sample points at uniform distances
    new_distances = np.arange(0, max_distance, step)

    # Interpolate x, y, z coordinates as functions of distance
    x_interp = interp1d(distances, [point[0] for point in data], kind='linear')
    y_interp = interp1d(distances, [point[1] for point in data], kind='linear')
    z_interp = interp1d(distances, [point[2] for point in data], kind='linear')

    # Sample the interpolation functions at the new distances
    new_x = x_interp(new_distances)
    new_y = y_interp(new_distances)
    new_z = z_interp(new_distances)

    return np.vstack((new_x, new_y, new_z)).T

import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def read_tum_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 8:
                x, y, z = map(float, parts[1:4])
                data.append((x, y, z))
    return data

def calculate_distances(data):
    distances = [0.0]
    for i in range(1, len(data)):
        prev_point = data[i - 1]
        curr_point = data[i]
        distance = np.sqrt((curr_point[0] - prev_point[0])**2 + (curr_point[1] - prev_point[1])**2 + (curr_point[2] - prev_point[2])**2)
        distances.append(distances[-1] + distance)
    return distances

def resample_path_uniformly(data, step):
    distances = calculate_distances(data)
    max_distance = distances[-1]

    # Create new sample points at uniform distances
    new_distances = np.arange(0, max_distance, step)

    # Interpolate x, y, z coordinates as functions of distance
    x_interp = interp1d(distances, [point[0] for point in data], kind='linear')
    y_interp = interp1d(distances, [point[1] for point in data], kind='linear')
    z_interp = interp1d(distances, [point[2] for point in data], kind='linear')

    # Sample the interpolation functions at the new distances
    new_x = x_interp(new_distances)
    new_y = y_interp(new_distances)
    new_z = z_interp(new_distances)

    return np.vstack((new_x, new_y, new_z)).T

def plot_2d_xy_with_nn_distances(data_a_np, data_output_np, time_window_proportion=1.0, save_name='output'):

    
    distances = np.zeros(len(data_output_np))
    pairs = []

    len_a = len(data_a_np)
    len_o = len(data_output_np)
    for i, point in enumerate(data_output_np[:, :2]):
        j = int(i * len_a / len_o)
        if j >= len_a:
            j = len_a - 1
        j_min = max(0, j - int(len_a * time_window_proportion / 2))
        j_max = min(len_a, j + int(len_a * time_window_proportion / 2))
        window = data_a_np[j_min:j_max, :2]
        all_distances = np.sqrt((window[:, 0] - point[0])**2 + (window[:, 1] - point[1])**2)
        min_idx = np.argmin(all_distances)
        distance = all_distances[min_idx]
        distances[i] = distance
        pairs.append([point, window[min_idx]])

        # distance = np.sqrt((data_a_np[j, 0] - point[0])**2 + (data_a_np[j, 1] - point[1])**2)
        # distances[i] = distance
        # pairs.append([point, data_a_np[j]])
    
    # print the abosolute error, mean error, and standard deviation, and rmse
    # print("Absolute error: ", np.sum(distances))
    print(f"Mean error: {np.mean(distances):.3f}")
    # print("Standard deviation: ", np.std(distances))
    print(f"RMSE: {np.sqrt(np.mean(distances**2)):.3f}")
    
    if True:
        # Create a 2D plot for the XY plane using nearest-neighbor distances for coloring
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot path 'a' in gray, use --
        ax.plot(data_a_np[:, 0], data_a_np[:, 1], color='gray', linewidth=2, label='Ground Truth', linestyle='--')
        
        # Scatter plot of 'output' colored by nearest-neighbor distance within window
        scatter = ax.scatter(data_output_np[:, 0], data_output_np[:, 1], c=distances, cmap='jet', s=3)

        # Colorbar to indicate error magnitude
        # cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15)
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.05)

        # Connect the nearest-neighbor pairs with lines
        # for pair in pairs:
        #     ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], color='red', linewidth=1)

        # don't scatter plot, but connect the points and show the color of the bar as the same as the scatter plot
        # for i in range(len(data_output_np)-1):
        #     color_value = plt.cm.jet( (distances[i] + distances[i+1]) / (2*np.max(distances)) )
        #     ax.plot([data_output_np[i, 0], data_output_np[i+1, 0]], [data_output_np[i, 1], data_output_np[i+1, 1]], color=color_value, linewidth=3)
        
        # Labels and legend
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        # ax.set_title('Error Mapped onto Path')
        ax.axis('equal')  # Ensure equal scaling for X and Y to preserve aspect ratio
        ax.legend()

        plt.savefig(os.path.join(base_path, f'{save_name}.pdf'))
        plt.show()


# Example usage
if __name__ == "__main__":
    # Paths to the TUM files
    # name = 'lego'
    # name = 'dlo'
    # name = 'path_aft_pgo_lc'
    # name = 'odom_wo_lc'
    # name = 'scdlo'
    name = 'cartographer'

    base_path = f'/home/liolc/res/{name}'

    file_path_a = os.path.join(base_path, 'ground_truth.txt')
    file_path_output = os.path.join(base_path, 'estimated_path.txt')

    # Read the TUM files
    data_a = read_tum_file(file_path_a)
    data_output = read_tum_file(file_path_output)

    # Resample the paths uniformly
    uniform_distance = 0.05  # Change this value as needed
    uniform_path_a = resample_path_uniformly(data_a, uniform_distance)
    uniform_distance = 0.1
    uniform_path_output = resample_path_uniformly(data_output, uniform_distance)

    plot_2d_xy_with_nn_distances(uniform_path_a, uniform_path_output, time_window_proportion=0.005, save_name=name)