import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon


class BodyPart():
  def __init__(self,joint_indexes = np.array([0]), angle_offset = 0,vector_joint_index = np.array([0, 1]), joint_origin_index= np.array([0]), parent_origin= np.array([0, 0]),  image_dims= np.array([0, 0]), shapely_fn=Polygon, offset = np.array([[0,0], [0,0]])):
    # set initially
    self.joint_indexes= joint_indexes
    self.joint_origin_index = joint_origin_index
    self.vector_joint_index = vector_joint_index
    self.angle_offset = angle_offset
    # offset = (left padding, right_padding, top padding, bottom padding)
    self.offset = offset
    self.confidence_threshold = 0.3
    self.image_dims = image_dims
    self.parent_origin = parent_origin
    self.shapely_fn = shapely_fn
   
    # updated
    self.part_detected = False
    self.part_image = None
    self.bounding_box = None
    self.joints = None
    
    self.joint_origin = None
    self.outline = None
    self.vector = None
    
    return
  def trackPart(self, landmarks, parent_image):

    confidence = np.array([point.visibility for point in landmarks.landmark])[self.joint_indexes]
    self.part_detected = (confidence > self.confidence_threshold).all()
    
    if(self.part_detected):

      self.joints = np.array([[point.x, point.y] for point in landmarks.landmark])[self.joint_indexes]
      self.joints *= self.image_dims
      self.joints = self.joints.astype('int64')
      
      
      # self.outline = np.array(list(self.shapely_fn(self.joints).buffer(100).exterior.coords)).astype('int64')
      # MultiPolygon

      self.outline = self.shapely_fn(self.joints).buffer(100)
      new_outline = []
      if(type(self.outline) == type(MultiPolygon(Polygon(self.joints), Polygon(self.joints)))):
        new_outline = []
        for polygon in self.outline:
          new_outline = new_outline + list(polygon.exterior.coords)
        self.outline = np.array(new_outline, dtype='int64')
      else:
        self.outline = np.array(list(self.shapely_fn(self.joints).buffer(100).exterior.coords)).astype('int64')

        
 
     
      joint_origin_list = np.array([[point.x, point.y] for point in landmarks.landmark])[self.joint_origin_index]
      joint_origin_list *= self.image_dims
      self.joint_origin  = np.array([np.sum(joint_origin_list[:, 0]), np.sum(joint_origin_list[:, 1])]) / self.joint_origin_index.shape[0]
      

      vector_joints = np.array([[point.x, point.y] for point in landmarks.landmark])[self.vector_joint_index]
      vector_joints *= self.image_dims
      self.vector = vector_joints[1] - vector_joints[0]
  

      self.boundingBox()
      self.cropImage(parent_image)

    else:
      self.part_image = None
      self.bounding_box = None
      self.joints = None
      self.joint_origin = None
      self.outline = None

  def boundingBox(self):
 
    self.bounding_box = np.array([
      [np.min(self.outline[:, 0]), np.min(self.outline[:, 1])], 
      [np.max(self.outline[:, 0]), np.max(self.outline[:, 1])]])

  
  def cropImage(self, parent_image):

    
    top_left= self.bounding_box[0] - self.offset[0]
    top_left = top_left.astype('int64')
    
    if(top_left[0] < 0.0):
      top_left[0] = 0.0
    if(top_left[1] < 0.0):
      top_left[1] = 0.0

    
    bottom_right = self.bounding_box[1] + self.offset[1]
    self.part_image = np.copy(parent_image[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]])
    

    self.joint_origin -= np.array([top_left[0], top_left[1]])
    self.joints -= np.array([top_left[0], top_left[1]])
    self.outline -= np.array([top_left[0], top_left[1]]).astype('int64')

    mask = np.zeros_like(self.part_image)
    cv2.fillPoly(mask, pts = [self.outline], color =(1,1,1))
    self.part_image *= mask

  def display(self,):
    cv2.imshow('MediaPipe Pose', self.part_image)
  
  def scale(self, scale_factor):

    new_size = np.array(self.part_image.shape) * scale_factor
    new_size[0], new_size[1] = np.array([new_size[1], new_size[0]]).astype('int64')
    if(new_size[0] == 0 or new_size[1] == 0):
      new_size = np.array([1, 1])
    self.part_image = cv2.resize(self.part_image, tuple(new_size.astype('int64'))[0:2])
    self.joint_origin  *= scale_factor
    self.joints = (self.joints*scale_factor).astype('int64')
    self.outline = (self.outline * scale_factor).astype('int64')

  
    return
  def debugDisplay(self,):
    for point in self.joints:
        cv2.circle(self.part_image, tuple(point - self.bounding_box[0] + self.offset[0]), 10, (255, 0, 0))
 
class Body():
  def __init__(self, frame_width, frame_height):
    # initialize with these
    # def __init__(self,joint_indexes, joint_origin_index, parent_origin,  image_dims, shapely_fn, offset = np.array([[0,0], [0,0]])):

    self.canvas_width = 1000
    self.canvas_height = 2000

    self.head_image = cv2.imread('/Users/hima/Desktop/body_seg/antannae.png')
    self.head_image = cv2.resize(self.head_image, tuple([int(self.head_image.shape[1] * 0.2), int(self.head_image.shape[0] * 0.2)]))

    
    self.Head = BodyPart(
      joint_indexes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), #joint_indexes
      joint_origin_index=np.array([10, 9]).astype('int64'), #joint_origin_index
      vector_joint_index=np.array([10, 9]).astype('int64'),
      parent_origin=np.array([self.canvas_height // 3, self.canvas_width // 2]).astype('int64'), #parent_origin
      image_dims=np.array([frame_width, frame_height]), #image_dims
      shapely_fn=Polygon, #shapely_fn
      offset=np.array([[40, 200], [40, 200]]).astype('int64'))
  
    self.LeftArm = BodyPart(
      joint_indexes=np.array([12, 14, 16, 18, 20, 22]), 
      joint_origin_index=np.array([12]),
      vector_joint_index = np.array([12, 14]),
      angle_offset = 180 + 35,
      parent_origin=np.array([self.canvas_height // 2, self.canvas_width // 3]),
      image_dims=np.array([frame_width, frame_height]),
      shapely_fn=LineString,
      offset=np.array([[100, 30], [100, 30]]).astype('int64'))
  
    self.RightArm = BodyPart(
      joint_indexes=np.array([11, 13, 15, 17, 19, 21]), 
      joint_origin_index=np.array([11]),
      angle_offset = -35,
      vector_joint_index = np.array([11, 13]),
      parent_origin=np.array([self.canvas_height//2 , self.canvas_width // 3 * 2]),
      image_dims=np.array([frame_width, frame_height]),
      shapely_fn=LineString)
    
    self.Torso = BodyPart(
      joint_indexes=np.array([12, 11, 23, 24]), 
      joint_origin_index=np.array([11, 12, 24, 23]),
      vector_joint_index = np.array([11, 12]),
      parent_origin=np.array([self.canvas_height // 2, self.canvas_width // 2]),
      image_dims=np.array([frame_width, frame_height]),
      shapely_fn=Polygon)
    self.LeftLeg = BodyPart(
      joint_indexes=np.array([24, 26, 28, 30]),
      joint_origin_index=np.array([24]),
      vector_joint_index = np.array([24,26]),
      parent_origin=np.array([self.canvas_height //3 * 2, self.canvas_width // 3]),
      image_dims=np.array([frame_width, frame_height]),
      angle_offset = 180 + 45,
      shapely_fn=LineString)
    self.RightLeg = BodyPart(
      joint_indexes=np.array([23, 25, 27, 29]),
      joint_origin_index=np.array([23]),
      vector_joint_index=np.array([23, 25]),
      parent_origin=np.array([self.canvas_height//3 * 2 , self.canvas_width // 3 * 2]),
      image_dims=np.array([frame_width, frame_height]),
      angle_offset = -45,
      shapely_fn=LineString)
    
    # self.partList = [self.Head, self.LeftArm, self.RightArm, self.Torso]
    self.partList = [self.Head, self.LeftArm, self.RightArm, self.LeftLeg, self.RightLeg]
    # self.partList = [self.LeftArm]

  
    # update over time
    self.image = np.zeros((self.canvas_height,self.canvas_width,3), np.uint8)
  def place_header(self):
    rev = np.array([self.head_image.shape[0], self.head_image.shape[1]])
    
    top_left = self.Head.parent_origin - rev / 2
    bottom_right = top_left + rev
    self.image[int(top_left[0]):int(bottom_right[0]), int(top_left[1]):int(bottom_right[1])] = self.head_image


  def update(self, camera_input, landmarks):
    self.place_header()
    for part in self.partList:
      part.trackPart(landmarks, camera_input)
    
    for part in self.partList:
      if(part.part_detected):
        part.scale(0.7)
        left_arm_offset_y = int(part.joint_origin[0])
        left_arm_offset_x = int(part.joint_origin[1])

        new_image = np.zeros_like(self.image)
        x1 = part.parent_origin[0] - left_arm_offset_x
        x2 = part.parent_origin[0] - left_arm_offset_x + part.part_image.shape[0]
        y1 = part.parent_origin[1] - left_arm_offset_y
        y2 = part.parent_origin[1] - left_arm_offset_y + part.part_image.shape[1]
        if(0 < x1 < new_image.shape[0] and 0 < x2 < new_image.shape[0] and
           0 < y1 < new_image.shape[1] and 0 < y2 < new_image.shape[1]):

          new_image[x1:x2, y1:y2] = part.part_image

          new_image = self.rotatePart(part, new_image)
  
          self.addPart(new_image)

        else:
          print("wrong dims!", x1, x2, y1, y2)
  def rotatePart(self, part, new_image):
    part_center =(part.parent_origin[1],  part.parent_origin[0])
    angle = np.arctan2(part.vector[1], part.vector[0])* 180/ np.pi + part.angle_offset
   
  
    M = cv2.getRotationMatrix2D((int(part_center[0]), int(part_center[1])), angle, 1.0)
    new_image = cv2.warpAffine(new_image, M, (new_image.shape[1], new_image.shape[0]))
    return new_image


  def addPart(self, new_image):

    img2gray = cv2.cvtColor(new_image,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    self.image = cv2.bitwise_and(self.image,self.image,mask = mask_inv)
    self.image += new_image

    return

  def display(self):
  
    cv2.imshow("body", self.image)
    self.image *= 0


# ----------------- main
cap = cv2.VideoCapture(0)
person = None

with mp_holistic.Holistic(
  
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