import cv2
import numpy as np
import pickle

class Serialiser:
    def save_to_file(self, keypoints, descriptors, filename):
        i = 0
        temp_array = []
        for point in keypoints:
            temp = (point.pt[0], point.pt[1], point.size, point.angle, point.response, point.octave,
            point.class_id, descriptors[i])
            temp_array.append(temp)
            i = i + 1
        pickle.dump(temp_array, open(filename + ".p", "wb"))


    def load_from_file(self, filename):
        keypoints = []
        descriptors = []
        temp = pickle.load( open(filename, "rb"))
        i = 0
        for i in range(len(temp)):
            temp_feature = cv2.KeyPoint(x=temp[i][0], y=temp[i][1], _size=temp[i][2], _angle=temp[i][3], _response=temp[i][4], _octave=temp[i][5], _class_id=temp[i][6])
            temp_descriptor = temp[i][7]
            keypoints.append(temp_feature)
            descriptors.append(temp_descriptor)
            i = i + 1
        return keypoints, np.array(descriptors)
