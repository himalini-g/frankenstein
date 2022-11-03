import cv2

image = cv2.imread('/Users/hima/Desktop/body_seg/antannae.png')

image = cv2.resize(image, tuple([int(image.shape[1] * 0.5), int(image.shape[0] * 0.5)]))
cv2.imshow("antannae", image)
cv2.waitKey(10000)