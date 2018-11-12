import cv2
import numpy as np
import pickle
from src.serialiser import Serialiser

test_image = cv2.imread('cane-toad1.jpg', 0)
template_image = cv2.imread('cane-toad-template.jpg', 0)

# start orb detection
orb = cv2.ORB_create()

# keypoints and descriptors
kp1, des1 = orb.detectAndCompute(test_image, None)
kp2, des2 = orb.detectAndCompute(template_image, None)
print(p.pt() for p in kp1)
print(des1)
'''# for serializer_old
serialiser = Serialiser()
serialiser.save_to_file(kp1, des1, "test.txt")
keypoints, descriptors = serialiser.load_from_file("test.txt")
'''
#for serializer
serialiser = Serialiser()

serialiser.save_to_file(kp1, des1, "keypoints_database1")
serialiser.save_to_file(kp2, des2, "keypoints_database2")


#Retrieve Keypoint Features
keyp1, desc1 = serialiser.load_from_file("keypoints_database1.p")
keyp2, desc2 = serialiser.load_from_file("keypoints_database2.p")

#print(descriptors)
#print(des2)

#print(keypoints)

# pattern matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
# matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
matches1 = bf.match(des1, trainDescriptors = des2)
matches1 = sorted(matches1, key = lambda x: x.distance)
matches2 = bf.match(desc1, trainDescriptors = des2)
matches2 = sorted(matches2, key = lambda x: x.distance)
img3 = cv2.drawMatches(test_image, kp1, template_image, kp2, matches1[:20], None, flags=2)
img4 = cv2.drawMatches(test_image, keyp1, template_image, kp2, matches2[:20], None, flags=2)


cv2.imshow('orb-image',img3)
cv2.imshow('orb-image-saved-then-loaded-datapoints',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
