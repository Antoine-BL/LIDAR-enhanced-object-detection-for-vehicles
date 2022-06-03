import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset

import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  plt.figure(figsize=(25, 20))
  ax = plt.subplot(*layout)

  for camera_label in camera_labels:
    # Draw the camera labels.
    # Iterate over the individual labels.
    for label in camera_label.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')

def extract_frame_features(data):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    image = frame.images[0].image
    labels = [camera_labels.labels for camera_labels in frame.camera_labels if camera_labels.name == 1][0]
    return (image, labels)

def extract_and_serialize_frame(data):
    (image, labels) = extract_frame_features(data)
    image_shape = tf.io.decode_jpeg(image).shape
    clazz = np.array([label.type for label in labels])
    center_x = np.array([label.box.center_x for label in labels])
    center_y = np.array([label.box.center_y for label in labels])
    height = np.array([label.box.length for label in labels])
    width = np.array([label.box.width for label in labels])

    feature = {
        'width': _int64_feature(image_shape[0]),
        'height': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'raw_image': _bytes_feature(image),
        'class': _int64List_feature(clazz),
        'box_center_y': _floatList_feature(value=center_y),
        'box_center_x': _floatList_feature(value=center_x),
        'box_width': _floatList_feature(value=width),
        'box_height': _floatList_feature(value=height)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64List_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _floatList_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def deserialize_example(record):
    tfrecord_format = {
        'width': tf.io.FixedLenFeature((), dtype=tf.int64),
        'height': tf.io.FixedLenFeature((), dtype=tf.int64),
        'depth': tf.io.FixedLenFeature((), dtype=tf.int64),
        'raw_image':  tf.io.FixedLenFeature((), dtype=tf.string),
        'class': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, default_value=-1, allow_missing=True),
        'box_center_x': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, default_value=-1, allow_missing=True),
        'box_center_y': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, default_value=-1, allow_missing=True),
        'box_width': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, default_value=-1, allow_missing=True),
        'box_height': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, default_value=-1, allow_missing=True)
    }

    feature_tensor = tf.io.parse_single_example(record, tfrecord_format)
    return feature_tensor

def parse_label(label):
    box = label.box
    {
        'class': label.type,
        'center_x': box.center_x,
        'center_y': box.center_y,
        'width': box.width,
        'height': box.height
    }