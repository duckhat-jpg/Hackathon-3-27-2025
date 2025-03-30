#imports
import cv2

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

import pygame

import tensorflow as tf
import tensorflow_hub as hub

matplotlib.use('TkAgg')

model_name = "movenet_lightning"
module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
input_size = 192

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

MIN_CROP_KEYPOINT_SCORE = 0.2

#imported from tensorflow movenet
def init_crop_region(image_height, image_width):
  if image_width > image_height:
    box_height = image_width / image_height
    box_width = 1.0
    y_min = (image_height / 2 - image_width / 2) / image_height
    x_min = 0.0
  else:
    box_height = 1.0
    box_width = image_height / image_width
    y_min = 0.0
    x_min = (image_width / 2 - image_height / 2) / image_width

  return {
    'y_min': y_min,
    'x_min': x_min,
    'y_max': y_min + box_height,
    'x_max': x_min + box_width,
    'height': box_height,
    'width': box_width
  }

#imported from tensorflow movenet
def torso_visible(keypoints):
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

#imported from tensorflow movenet
def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

#imported from tensorflow movenet
def determine_crop_region(
      keypoints, image_height,
      image_width):
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)

#imported from tensorflow movenet
def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    # Ensure image is 4D [batch_size, height, width, channels]
    if len(image.shape) == 5:
        image = tf.squeeze(image, axis=1)  # Remove the extra dimension
    
    boxes = [[crop_region['y_min'], crop_region['x_min'],
            crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image

#imported from tensorflow movenet
def run_inference(movenet, image, crop_region, crop_size):
  # Get height and width from the image shape
  image_height, image_width, _ = image.shape[:3]
  
  input_image = crop_and_resize(
    tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
  # Run model inference.
  keypoints_with_scores = movenet(input_image)
  # Update the coordinates.
  for idx in range(17):
    keypoints_with_scores[0, 0, idx, 0] = (
        crop_region['y_min'] * image_height +
        crop_region['height'] * image_height *
        keypoints_with_scores[0, 0, idx, 0]) / image_height
    keypoints_with_scores[0, 0, idx, 1] = (
        crop_region['x_min'] * image_width +
        crop_region['width'] * image_width *
        keypoints_with_scores[0, 0, idx, 1]) / image_width
  return keypoints_with_scores

#imported from tensorflow movenet
def resize_frame(frame, target_size):
    height, width = frame.shape[:2]
    scale = target_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(frame, (new_width, new_height))

#imported from tensorflow movenet
def movenet(input_image):
    model = module.signatures['serving_default']

    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

#imported from tensorflow movenet
def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

#improved from tensorflow movenet to work with constant webcam input
def draw_prediction_on_image(image, keypoints_with_scores, ax=None, line_segments=None, scat=None):
  
  height, width, _ = image.shape

  #for first frame
  if ax is None or line_segments is None or scat is None:
      fig, ax = plt.subplots(figsize=(12, 12))
      ax.margins(0)
      ax.set_yticklabels([])
      ax.set_xticklabels([])
      plt.axis('off')
      
      line_segments = LineCollection([], linewidths=(4), linestyle='solid')
      ax.add_collection(line_segments)
      scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
      im = ax.imshow(image)
  else:
      #remove previous lines
      for collection in ax.collections:
          collection.remove()
      
      #add lines
      line_segments = LineCollection([], linewidths=(4), linestyle='solid')
      ax.add_collection(line_segments)
      scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
      
      #for when first image has not been made yet
      if hasattr(ax, 'images') and ax.images:
          ax.images[0].set_data(image)
      else:
          ax.imshow(image)

  #update all keypoints
  (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(
      keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  scat.set_offsets(keypoint_locs)

  plt.pause(0.001)

  return ax

#helper to get angles
def get_angle(point1, point2, point3):
    #convert to numpy array
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    
    #get vectors
    vec1 = point1 - point2
    vec2 = point3 - point2
    
    #using cos(theta) = (a dot b) / (magnitude(a) times magnitude(b))
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    #do arccos of that
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    #get that angle in degrees
    angle = np.degrees(angle)
    return angle

#main function
def main():
    #capture webcam
    vc = cv2.VideoCapture(0)
    first_frame = False
    squat_down = False
    squat_up = False

    squat_count = 0

    plt.ion()
    figure, ax = plt.subplots(figsize=(12, 12))
    ax.axis("off")
    line_segments = LineCollection([], linewidths=4, linestyle='solid')
    ax.add_collection(line_segments)
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    pygame.mixer.init()
    sound = pygame.mixer.Sound("C:/Users/lufre/OneDrive/Desktop/College/Hackathon 3-27-2025/bell-sound.mp3")
    sound.set_volume(0.5)

    if (not (vc.isOpened())):
        print("open webcam error")
        vc.release()
    else:
        while(True):
            ret, frame = vc.read()
            if (not ret):
                break
            
            #convert to rgb format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_height, frame_width = frame.shape[:2]
            
            #get crop region on first open
            if first_frame == False:
                crop_region = init_crop_region(frame_height, frame_width)
                sound.play()
                pygame.time.delay(int(sound.get_length() * 1000))
                sound.play()
                pygame.time.delay(int(sound.get_length() * 1000) / 5)
                sound.play()
                pygame.time.delay(int(sound.get_length() * 1000) / 5)
                sound.play()
            #convert for usage in keypoints
            frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
            
            keypoints_with_scores = run_inference(movenet, frame_tensor, crop_region, crop_size=[input_size, input_size])

            keypoints = {
                'left_hip': keypoints_with_scores[0, 0, KEYPOINT_DICT['left_hip']],
                'right_hip': keypoints_with_scores[0, 0, KEYPOINT_DICT['right_hip']],
                'left_knee': keypoints_with_scores[0, 0, KEYPOINT_DICT['left_knee']],
                'right_knee': keypoints_with_scores[0, 0, KEYPOINT_DICT['right_knee']],
                'left_ankle': keypoints_with_scores[0, 0, KEYPOINT_DICT['left_ankle']],
                'right_ankle': keypoints_with_scores[0, 0, KEYPOINT_DICT['right_ankle']]
            }
            
            left_knee_angle = get_angle(
                keypoints['left_hip'][:2],
                keypoints['left_knee'][:2],
                keypoints['left_ankle'][:2]
            )
            
            right_knee_angle = get_angle(
                keypoints['right_hip'][:2],
                keypoints['right_knee'][:2],
                keypoints['right_ankle'][:2]
            )

            if (right_knee_angle <= 100 and left_knee_angle <= 100 and squat_down == False):
              squat_count += 1
              squat_down = True
              squat_up = False
              sound.play()
              
            
            if (right_knee_angle >= 120 and left_knee_angle >= 120 and squat_up == False):
              squat_down = False
              squat_up = True
              sound.play()
              pygame.time.delay(int(sound.get_length() * 1000))
              sound.play()

            print(f"\nLeft Knee Angle: {left_knee_angle:.1f}°")
            print(f"Right Knee Angle: {right_knee_angle:.1f}°")
            print(f"Squat Count: {squat_count}")
            
            ax = draw_prediction_on_image(frame, keypoints_with_scores, ax, line_segments, scat)

            #get next crop region
            crop_region = determine_crop_region(keypoints_with_scores, frame_height, frame_width)

            first_frame = True

            plt.pause(0.001)

            if (squat_count == 10):
              break

            #quit out
            if plt.get_fignums() == [] or cv2.waitKey(1) & 0xFF == ord('q'):
                break

    #make sure to rlease everything at the end
    vc.release()
    cv2.destroyAllWindows()
    plt.close()

main()
