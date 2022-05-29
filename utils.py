import os
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

def frame_reader(target_dir):
    file_paths = os.listdir(target_dir)
    for file_path in file_paths:
        if not os.isfile(file_path):
            continue
        for frame in _file_frame_reader(file_path):
            yield frame

def _file_frame_reader(file_path):
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    for data in dataset:
        frame