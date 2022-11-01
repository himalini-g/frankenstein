# For webcam input:
# plan 
# get image 
# paste image
# paste all the images in a rough location
# 
from re import L
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

cap = cv2.VideoCapture(0)
class BodyPart():
  def __init__(self,joint_indexes, joint_origin_index, parent_origin,  image_dims, offset = np.array([[0,0], [0,0]])):
    # set initially
    self.joint_indexes= joint_indexes
    self.joint_origin_index = joint_origin_index
    # offset = (left padding, right_padding, top padding, bottom padding)
    self.offset = offset
    self.confidence_threshold = 0.3
    self.image_dims = image_dims
    self.parent_origin = parent_origin
   
    # updated
    self.part_detected = False
    self.part_image = None
    self.bounding_box = None
    self.joints = None
    self.joint_origin = None
    return
  def trackPart(self, landmarks, parent_image):

    confidence = np.array([point.visibility for point in landmarks.landmark])[self.joint_indexes]
    self.part_detected = (confidence > self.confidence_threshold).all()
    
    if(self.part_detected):

      self.joints = np.array([[point.x, point.y] for point in landmarks.landmark])[self.joint_indexes]
      self.joints *= self.image_dims
      self.joints = self.joints.astype('int64')
     
      self.joint_origin  = np.array([np.sum(self.joints[:, 0][self.joint_origin_index]), np.sum(self.joints[:, 1][self.joint_origin_index])]) / self.joint_origin_index.shape[0]
    

      self.boundingBox()
      self.cropImage(parent_image)
      self.debugDisplay()
    else:
      self.part_image = None
      self.bounding_box = None
      self.joints = None
      self.joint_origin = None

  def boundingBox(self):
    self.bounding_box = np.array([
      [np.min(self.joints[:, 0]), np.min(self.joints[:, 1])], 
      [np.max(self.joints[:, 0]), np.max(self.joints[:, 1])]])
  
  def cropImage(self, parent_image):
    top_left= self.bounding_box[0] - self.offset[0]
    self.joint_origin -= top_left
    if(top_left[0] < 0):
      top_left[0] = 0
    if(top_left[1] < 0):
      top_left[1] = 0

    
    bottom_right = self.bounding_box[1] + self.offset[1]
    self.part_image = parent_image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

  def display(self,):
    cv2.imshow('MediaPipe Pose', self.part_image)
  
  def scale(self, scale_factor):

    new_size = np.array(self.part_image.shape) * scale_factor
    new_size[0], new_size[1] = new_size[1], new_size[0]
  
  
    self.part_image = cv2.resize(self.part_image, tuple(new_size.astype('int64'))[0:2])
    self.joint_origin  *= scale_factor
  
    return
  def debugDisplay(self,):
    for point in self.joints:
        cv2.circle(self.part_image, tuple(point - self.bounding_box[0] + self.offset[0]), 10, (255, 0, 0))
    cv2.circle(self.part_image, tuple(self.joint_origin.astype('int64') - self.bounding_box[0] + self.offset[0]), 10, (0, 255, 0))


class Body():
  def __init__(self, frame_width, frame_height):
    # initialize with these
    self.Head = BodyPart(
      np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 
      np.array([10, 9]).astype('int64'),
      np.array([700, 700]).astype('int64'),
      np.array([frame_width, frame_height]), 
      offset=np.array([[40, 200], [40, 200]]).astype('int64'))
  
    self.LeftArm = BodyPart(
      np.array([12, 14, 16, 18, 20, 22]), 
      [12],
      [500, 500],
      np.array([frame_width, frame_height]))
  
    # self.RightArm = BodyPart(
    #   np.array([11, 13, 15, 17, 29, 21]), 
    #   [11],
    #   [300, 200],
    #   np.array([frame_width, frame_height]))
   
    # self.Torso = BodyPart(
    #   np.array([12, 11, 23, 24]), 
    #   [11, 12, 24, 23],
    #   [200, 250],
    #   np.array([frame_width, frame_height]))
    self.partList = [self.Head]
    # self.partList = [self.Head, self.LeftArm, self.RightArm, self.Torso]

    self.canvas_width = 2000
    self.canvas_height = 2000

    # update over time
    self.image = np.zeros((self.canvas_height,self.canvas_width,3), np.uint8)

  def update(self, camera_input, landmarks):
    self.Head.trackPart(landmarks, camera_input)
 
    
    if(self.Head.part_detected):
      self.Head.scale(0.3)
  
      image_offset_x = int(self.Head.part_image.shape[0])
      image_offset_y = int(self.Head.part_image.shape[1]//2)

      self.image[
        self.Head.parent_origin[0] - image_offset_x :self.Head.parent_origin[0] - image_offset_x + self.Head.part_image.shape[0], 
        self.Head.parent_origin[1] - image_offset_y:self.Head.parent_origin[1] -  image_offset_y + self.Head.part_image.shape[1]] = self.Head.part_image
 
 
  def display(self):
  
    cv2.imshow("body", self.image)
    self.image *= 0

person = None

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
      if(person == None):
        person = Body(frame_width, frame_height)
      person.update(image, results.pose_landmarks)

    if(person != None):
      person.display()
      
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()