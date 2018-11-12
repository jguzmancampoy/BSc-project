import cv2
import numpy as np

class Serialiser:
    def to_string(array):
        return ","

    def save_to_file(self, keypoints, descriptors, filename):
        i = 0
        file = open(filename, "w+")
        for point in keypoints:
            file.write("%d,%d,%d,%d,%d,%d,%d" % (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave, point.class_id))
            file.write(',')
            file.write(np.array2string(descriptors[i], separator="|").replace("\n", '').replace(' ', ''))
            file.write("\n")
            i = i + 1

    def load_from_file(self, filename):
        file = open(filename, "r")
        keypoints = []
        descriptors = np.array([])
        for l in file:
            line = l.strip().split(',')
            descriptor = np.array(list(map(int, line[7].replace('[', '').replace(']', '').split("|"))))
            keypoint = cv2.KeyPoint(x=float(line[0]), y=float(line[1]),_size=float(line[2]), _angle=float(line[3]), _response=float(line[4]), _octave=int(line[5]), _class_id=int(line[6]))
            keypoints.append(keypoint)
            descriptors = np.append(descriptors, descriptor)
        return keypoints, descriptors
