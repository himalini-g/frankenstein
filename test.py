# For webcam input:
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

cap = cv2.VideoCapture(0)
class BodyPart():
  def __init__(self,landmark_indexes,offset = np.array([[0,0], [0,0]])):
    # set initially
    self.landmark_indexes= landmark_indexes
    # offset = (left padding, right_padding, top padding, bottom padding)
    self.offset = offset
    self.confidence_threshold = 0.3
    self.image_dims = None
    
    # updated
    self.part_detected = False
    self.part_image = None
    self.bounding_box = None
    self.joints = None
    return
  def trackPart(self, landmarks, parent_image):

    confidence = np.array([point.visibility for point in landmarks.landmark])[self.landmark_indexes]
    self.part_detected = (confidence > self.confidence_threshold).all()
    
    if(self.part_detected):

      self.joints = np.array([[point.x, point.y] for point in landmarks.landmark])[self.landmark_indexes]
      self.joints *= self.image_dims
      self.joints = self.joints.astype('int64')

      self.boundingBox()
      self.cropImage(parent_image)
    else:
      self.part_image = None
      self.bounding_box = None
      self.joints = None

  def boundingBox(self):
    self.bounding_box = np.array([
      [np.min(self.joints[:, 0]), np.min(self.joints[:, 1])], 
      [np.max(self.joints[:, 0]), np.max(self.joints[:, 1])]])
  
  def cropImage(self, parent_image):
    top_left= self.bounding_box[0] - self.offset[0]
    bottom_right = self.bounding_box[1] + self.offset[1]
    self.part_image = parent_image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
  
  def display(self,):
    cv2.imshow('MediaPipe Pose', self.part_image)
  
  def debugDisplay(self,):
    for point in self.joints:
        cv2.circle(self.part_image, tuple(point - self.bounding_box[0] + self.offset[0]), 10, (255, 0, 0))
class Body():
  def __init__(self):
    self.Head = BodyPart(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), offset=np.array([[40, 200], [40, 200]]))
    self.LeftArm = BodyPart(np.array([]))


    
Head = BodyPart(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
LeftArm = BodyPart(np.array())
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

      frame_height, frame_width = image.shape[:2]
      Head.image_dims = np.array([frame_width, frame_height])
      Head.trackPart(results.pose_landmarks, image)



    # cv2.imshow('MediaPipe Pose', image)
    if(Head.part_detected):
      Head.debugDisplay()
      Head.display()
      
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()