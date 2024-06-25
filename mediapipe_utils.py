import csv
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


MODEL_TYPE = 'lite'
MODEL_PATH = f'pose_landmarker_{MODEL_TYPE}.task'
MODE = VisionRunningMode.VIDEO
options = PoseLandmarkerOptions(
        # base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.GPU),
        base_options=BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU),
        output_segmentation_masks=True,
        running_mode=MODE,
        min_pose_detection_confidence=0.1)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def write_pose_video(inp_video, csv_file):
    fieldnames = ['0', '1', '2']
    with open(csv_file, 'w') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()

    cap = cv2.VideoCapture(inp_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
                # VIDEO
                pose_landmarker_result = landmarker.detect_for_video(mp_image, int(frame_i / fps * 1000))
                with open(csv_file, 'a') as file:
                    csv_writer  = csv.DictWriter(file, fieldnames=fieldnames)
                    if len(pose_landmarker_result.pose_landmarks):
                        info = {
                            '0': int(pose_landmarker_result.pose_landmarks[0][0].x*w),
                            '1': int(pose_landmarker_result.pose_landmarks[0][1].x*w),
                            '2': int(pose_landmarker_result.pose_landmarks[0][2].x*w)
                        }
                    else:
                        info = {
                            '0': None, '1': None, '2': None
                        }
                    csv_writer.writerow(info)
                    logging.info(info)

                frame_i = frame_i + 1

            else:
                break
        cap.release()
        cv2.destroyAllWindows()




        

