import os
import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R

def write_to_tum(file_path, data):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(' '.join(map(str, line)) + '\n')

def main(bag_file, odom_topic, ground_truth_topic, odom_out_path, gt_out_path):
    bag = rosbag.Bag(bag_file)
    odom_data = []
    ground_truth_data = []

    # Read odometry data
    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        timestamp = t.to_sec()
        x, y, z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
        qx, qy, qz, qw = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        odom_data.append([timestamp, x, y, z, qx, qy, qz, qw])

    # Read ground truth data, do not downsample
    for topic, msg, t in bag.read_messages(topics=[ground_truth_topic]):
        timestamp = t.to_sec()
        x, y, z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
        qx, qy, qz, qw = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        ground_truth_data.append([timestamp, x, y, z, qx, qy, qz, qw])
    
    print("Odom data length: ", len(odom_data))
    print("Ground truth data length: ", len(ground_truth_data))

    # Write to TUM format
    write_to_tum(odom_out_path, odom_data)
    write_to_tum(gt_out_path, ground_truth_data)

    bag.close()

if __name__ == "__main__":
    base_path = '/home/liolc/res/dlo'
    bag_file = os.path.join(base_path, 'my_experiment.bag')
    odom_topic = '/robot/dlo/odom_node/odom' # '/odometry/filtered'
    ground_truth_topic = '/gazebo/ground_truth/state'
    odom_out_path = os.path.join(base_path, 'estimated_path.txt')
    gt_out_path = os.path.join(base_path, 'ground_truth.txt')
    main(bag_file, odom_topic, ground_truth_topic, odom_out_path, gt_out_path)
