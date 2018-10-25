import cv2
import numpy as np
import os


path = '/Users/julioguzmancampoy/Desktop/OPENCV/'
toad1 = path + 'toad1.jpg'
toad2 = path + 'toad2.jpg'
template1 = path + 'template1.jpg'
template2 = path + 'template2.jpg'

#Loading/reading image as matrix
img_rgb = cv2.imread(toad1)
print(img_rgb)

#converting image to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#loading/reading template as matrix
template = cv2.imread(template1,0)
print(template)

#size of template (x,y)
w, h = template.shape[::-1]

#matching template using ccoeff_normed method and setting threshold at 0.9
match = cv2.matchTemplate(img_gray,template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(match >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,100,150), 2)

#visualising match
cv2.imshow('detected',img_rgb)

#closing image with esc key
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
