# import required packages
import cv2
import dlib
import argparse
import time
from google.colab.patches import cv2_imshow

# load input image
image = cv2.imread('/content/drive/MyDrive/IOT Project/known faces/Hritik Roshan.jpg')

if image is None:
    print("Could not read input image")
    exit()
    
# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1('/content/drive/MyDrive/IOT Project/MODELS/mmod_human_face_detector.dat')
start = time.time()

# apply face detection (hog)
faces_hog = hog_face_detector(image, 1)

end = time.time()
print("Execution Time (in seconds) :")
print("HOG : ", format(end - start, '.2f'))