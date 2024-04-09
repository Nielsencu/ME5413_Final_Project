import pandas as pd

# Assuming you've already read the CSV correctly
df = pd.read_csv('ground_truth_2.txt', delimiter=',')

# Convert the timestamp from nanoseconds to seconds
df['timestamp'] = df['%time'] / 1e9

# Select the necessary columns for the TUM format
tum_format_df = df[['timestamp',
                    'field.pose.pose.position.x',
                    'field.pose.pose.position.y',
                    'field.pose.pose.position.z',
                    'field.pose.pose.orientation.x',
                    'field.pose.pose.orientation.y',
                    'field.pose.pose.orientation.z',
                    'field.pose.pose.orientation.w']]

# Rename the columns to match TUM format (optional for clarity in output file)
tum_format_df.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

# Save the formatted data to a new TUM file
tum_format_df.to_csv('ground_truth_2.tum', header=False, index=False, sep=' ', float_format='%.9f')
