import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np
from shapely.geometry import Polygon, LineString


import ssl
ssl._create_default_https_context = ssl._create_unverified_context

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=2) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if(results.pose_landmarks != None):

        joints = np.array([[point.x, point.y] for point in results.pose_landmarks.landmark])
        frame_height, frame_width = image.shape[:2]
        
        image_dims = np.array([frame_width, frame_height])
        joints *= image_dims
        joints = joints.astype('int64')

        
        for point in joints:
            cv2.circle(image, tuple(point), 10, (255, 0, 0), 10)
        

        torso =  np.array([12, 11, 23, 24])
        rightArm = np.array([11, 13, 15, 17, 19, 21])
        outline = np.array(list(LineString(joints[rightArm]).buffer(60).exterior.coords), dtype=np.int32)
        # print(outline.shape)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, pts = [outline], color =(1,1,1))
        image = image * mask
        cv2.polylines(image,[outline],False, (0, 255, 0), 10)
        for point in outline:
            cv2.circle(image, tuple(point), 10, (0, 255, 0), 10)
        
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()