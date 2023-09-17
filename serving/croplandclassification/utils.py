import time
import math
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from .config import *
from .data import labeled_feature

def extract_features(row):
    target_features  = BANDS + FEATURES + [LABEL]
    feature_values =  labeled_feature(row.Lon, row.Lat, row.Target)
    # Create a dictionary to store each band as a feature
    feature_dict = {band: np.array(feature_values.get(band).getInfo()) for band in target_features}

    return feature_dict


# Function to convert a dictionary of features to TFRecord
def dict_to_tf_example(feature_dict):
    # Convert each feature to a TensorFlow Feature
    features = {}
    for band in (BANDS + FEATURES):
        features[band] = tf.train.Feature(float_list=tf.train.FloatList(value=feature_dict[band].flatten()))

    # Label feature
    features[LABEL] = tf.train.Feature(float_list=tf.train.FloatList(value=[feature_dict[LABEL].item()]))

    # Create a Features message using tf.train.Example
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example

# Process each row in the DataFrame
def process_rows(row, output_path):
    lon, lat, target = row['Lon'], row['Lat'], row['Target']
    feature_dict = extract_features(row)
    example = dict_to_tf_example(feature_dict)

    # Serialize the example to string
    tfrecord = example.SerializeToString()
   
    with tf.io.TFRecordWriter(f"{output_path}/{lon}_{lat}_tfrecord.tfrecord") as writer:
        writer.write(tfrecord)

def create_tfrecord(dataframe, output):
    total_rows = len(dataframe)
    for index, row in tqdm(dataframe.iterrows(), total=total_rows):
      process_rows(row, output)


def monitor_ee_tasks(task_list, check_interval=60):
    """
    Monitors a list of Earth Engine tasks until all tasks are completed or failed.

    Args:
        task_list (list): A list of Earth Engine tasks to monitor.
        check_interval (int): The interval in seconds between task status checks (default is 60 seconds).
    """
    # Check if the task list is empty
    if not task_list:
        print("No tasks to monitor.")
        return

    # Start the tasks
    for task in task_list:
        task.start()

    # Monitor the tasks
    while any(task.status()["state"] in ["READY", "RUNNING"] for task in task_list):
        print("Monitoring tasks...")
        time.sleep(check_interval)

    # Check the final status of each task
    for i, task in enumerate(task_list):
        if task.status()["state"] == "COMPLETED":
            print()
            print(f"Task {i + 1} completed successfully!\n")
            task_status = task.status()
            if "description" in task_status or "destination_uris" in task_status:
                print(f"File Description: {task_status['description']}\n")
                print(f"Destination url: {task_status['destination_uris']}\n")
        else:
            print(f"Task {i + 1} failed or was canceled.")
