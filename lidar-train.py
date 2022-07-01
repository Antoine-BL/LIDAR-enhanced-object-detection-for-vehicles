# %%
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.getcwd()))
import tensorflow as tf
import numpy as np
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import lidar_crop, save_darknet, label_in_polygon

# %%
TOO_DARK_THRESHOLD = 50

# %%
training_files = ['Data/lidar tests/training/' + file for file in os.listdir('Data/lidar tests/training')]
dataset = tf.data.TFRecordDataset(training_files, buffer_size=tf.constant(int(pow(10,6)), tf.int64), num_parallel_reads=16)
OUTPUT_PATH = 'Data/darknet-data'
img_id = len(os.listdir(f'{OUTPUT_PATH}/obj')) // 2

training_index = open('Data/darknet-data/training.txt', 'w')
print('Processing training data')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    image = frame.images[0]
    image = np.array(tf.io.decode_jpeg(image.image))

    if np.mean(image) < TOO_DARK_THRESHOLD:
        continue
    
    og_width = image.shape[1]
    og_height = image.shape[0]
    (image, labels) = lidar_crop(frame, image)
    labels = [label for label in labels if label_in_polygon(label, image, 0.25)]
    
    save_darknet(image, labels, OUTPUT_PATH, og_width, og_height, training_index, img_id)
    
    img_id += 1
    if img_id % 100 == 0:
        print(img_id)

training_index.close()

# %%
validation_files = ['Data/lidar tests/validation/' + file for file in os.listdir('Data/lidar tests/validation')]
dataset = tf.data.TFRecordDataset(validation_files, buffer_size=tf.constant(int(pow(10,6)), tf.int64), num_parallel_reads=16)

validation_index = open('Data/darknet-data/validation.txt', 'w')
img_id = len(os.listdir(f'{OUTPUT_PATH}/obj')) // 2

print('Processing validation data')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    image = frame.images[0]
    image = np.array(tf.io.decode_jpeg(image.image))

    if np.mean(image) < TOO_DARK_THRESHOLD:
        continue
    
    og_width = image.shape[1]
    og_height = image.shape[0]
    (image, labels) = lidar_crop(frame, image)
    
    save_darknet(image, labels, OUTPUT_PATH, og_width, og_height, validation_index, img_id)
    
    img_id += 1
    if img_id % 100 == 0:
        print(img_id)

validation_index.close()


