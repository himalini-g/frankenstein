# For webcam input:
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

cap = cv2.VideoCapture(0)
class BodyPart():
  def __init__(self,landmark_indexes, offset):
    self.landmark_indexes= landmark_indexes
    self.offset = offset
    self.part_detected = False
    self.landmarks = []
    self.part_image = None
    return
  def trackImage(self, landmarks):
    confidence = np.array([point.visibility for point in landmarks.landmark])[self.landmark_indexes]
    print(confidence)
    print((confidence > 0.3).all())

    
Head = BodyPart(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), offset = np.array([0, 0, 0, 0]))
with mp_pose.Pose(
  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation = True,
    static_image_mode = False) as pose:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
   
    if(type(results.segmentation_mask) == type(np.array([1.0]))):
      condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
      image = image * condition

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
    if(results.pose_landmarks != None):
      # print(results.pose_landmarks.landmark)
      Head.updateImage(results.pose_landmarks)
      points = np.array([[point.x, point.y] for point in results.pose_landmarks.landmark])[0:11]
      visibility = np.array([])
      frame_height, frame_width = image.shape[:2]
      points *= np.array([frame_width, frame_height])
      points = points.astype('uint64')
      # print(len(points))
    

      for point in points:
        # print(point)
        cv2.circle(image, tuple(point), 10, (255, 0, 0))

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()