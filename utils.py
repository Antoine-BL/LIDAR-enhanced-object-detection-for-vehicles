import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import  frame_utils

import matplotlib.pyplot as plt
from matplotlib import patches

import numpy as np
import math
import open3d as o3d
import alphashape
from PIL import Image
import cv2


CAMERA_NAME_FRONT = 1

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
    labels = [camera_labels.labels for camera_labels in frame.camera_labels if camera_labels.name == CAMERA_NAME_FRONT][0]
    return (image, labels)

def extract_and_serialize_frame(data):
    (image, labels) = extract_frame_features(data)
    return serialize_frame(image, labels)

def serialize_frame(image, labels):
    image_shape = tf.io.decode_jpeg(image).shape
    clazz = np.array([label.type for label in labels])
    center_x = np.array([label.box.center_x for label in labels])
    center_y = np.array([label.box.center_y for label in labels])
    height = np.array([label.box.width for label in labels])
    width = np.array([label.box.length for label in labels])

    feature = {
        'width': _int64_feature(image_shape[1]),
        'height': _int64_feature(image_shape[0]),
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

def label_in_polygon(label, image):
    top_left_x = int(math.ceil(label.box.center_x - label.box.length / 2))
    top_left_y = int(math.ceil(label.box.center_y - label.box.width / 2))
    bottom_right_x = int(math.ceil(label.box.center_x + label.box.length / 2))
    bottom_right_y = int(math.ceil(label.box.center_y + label.box.width / 2))
    cropped = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    nb_pixels = cropped.size
    nonzero = np.count_nonzero(cropped)
    return (nonzero / nb_pixels) > 0.5

SCALED_WIDTH = 800
SCALED_HEIGHT = 533
LABEL_VEHICLE = 1
LABEL_PEDESTRIAN = 2
SUPPORTED_LABELS = {
    1: 0,
    2: 1,
    4: 2
}

def lidar_crop(frame, image):
    (range_images, camera_projections, _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame,
    range_images,
    camera_projections,
    range_image_top_pose)
    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)
    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    all_points = np.append(points_all, points_all_ri2, axis=0)
    all_points_cp = np.append(cp_points_all, cp_points_all_ri2, axis=0)

    front_points = np.array([point for (point, cp) in zip(all_points, all_points_cp) if cp[0] == 1 or cp[3] == 1])
    front_cp_points = np.array([cp[:3] if cp[0] == 1 else cp[3:] for cp in all_points_cp if cp[0] == 1 or cp[3] == 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(front_points)
    _, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=4, num_iterations=1000)

    front_cp_points_no_road = np.delete(front_cp_points, inliers, axis=0)

    alpha_shape = alphashape.alphashape(front_cp_points_no_road[:,1:3], 0.02)
    int_coords = lambda x: np.array(x).round().astype(np.int32)

    try:
        exteriors = [int_coords(poly.exterior.coords) for poly in alpha_shape]
    except:
        exteriors = [int_coords(poly.exterior.coords) for poly in [alpha_shape]]

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.fillPoly(mask, exteriors, 1)
    image = cv2.bitwise_and(image, image, mask=mask)

    labels = [camera_labels.labels for camera_labels in frame.camera_labels if camera_labels.name == 1][0]
    labels = [label for label in labels if label.box.length > 50 and label.box.length > 50 and label.type in SUPPORTED_LABELS.keys()]

    return (image, labels)

def save_darknet(image, labels, output_path, og_width, og_height, index_file, img_id):
    img_width = SCALED_WIDTH
    img_height = SCALED_HEIGHT
    image = Image.fromarray(image)
    image = image.resize((img_width, img_height))
    image_name = f'example-{img_id}.jpg'
    image.save(f'{output_path}/obj/{image_name}')

    label_name = f'example-{img_id}.txt'
    with open(f'{output_path}/obj/{label_name}', 'w') as label_file:
        for label in labels:
            cx = label.box.center_x 
            cy = label.box.center_y
            w = label.box.length
            h = label.box.width
            cx /= og_width
            cy /= og_height
            w /= og_width
            h /= og_height
            clazz = SUPPORTED_LABELS[label.type]
            darknet_labels = f'{clazz} {cx} {cy} {w} {h}\n'
            label_file.write(darknet_labels)

    index_file.write(f'data/obj/{image_name}\n')

def label_in_polygon(label, image, threshold = 0.25):
    top_left_x = int(math.ceil(label.box.center_x - label.box.length / 2))
    top_left_y = int(math.ceil(label.box.center_y - label.box.width / 2))
    bottom_right_x = int(math.ceil(label.box.center_x + label.box.length / 2))
    bottom_right_y = int(math.ceil(label.box.center_y + label.box.width / 2))
    cropped = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    nb_pixels = cropped.size
    nonzero = np.count_nonzero(cropped)
    return (nonzero / nb_pixels) > threshold