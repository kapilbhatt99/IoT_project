import dlib
import scipy.misc
import numpy as np
import os
import imageio
import cv2
import datetime as dt
import time
import pyrebase
import pytz
import firebase

Config = {
  "apiKey": "AIzaSyBplgfDMaVZSDahuI2RwF24a7e2K4vVXcs",
  "authDomain": "videobase-dynamic-auth-system.firebaseapp.com",
  "databaseURL": "https://videobase-dynamic-auth-system.firebaseio.com",
  "projectId": "videobase-dynamic-auth-system",
  "storageBucket": "videobase-dynamic-auth-system.appspot.com",
  "messagingSenderId": "542414051699",
  "appId": "1:542414051699:web:043b564a6117971ac88d06"
}

firebase=pyrebase.initialize_app(Config)
storage=firebase.storage()

weights = '/content/drive/My Drive/ML project/MODELS/mmod_human_face_detector.dat'

# Get Face Detector from dlib
# This allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()
# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('/content/drive/My Drive/ML project/MODELS/shape_predictor_68_face_landmarks.dat')
# Get the face recognition model
# This is what gives us the face encodings (numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('/content/drive/My Drive/ML project/MODELS/dlib_face_recognition_resnet_model_v1.dat')
# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.5

# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    # Load image using scipy
    image = imageio.imread(path_to_image)
    # Detect faces using the face detector
    detected_faces = face_detector(image, 1)
    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

tz = pytz.timezone('Asia/Kolkata')

# Upload Function
def upload(paths_to_images,phone_number,expdate,name,sec_code,status,designation):
    # List of face encodings we have calulated
    face_encodings = []
    # List of image urls
    urls = []
    # Loop over images to get the encoding one by one
    for path_to_image in paths_to_images:
        # Get face encodings from the image
        face_encodings_in_image = get_face_encodings(path_to_image)
        # Make sure there's exactly one face in the image
        if len(face_encodings_in_image) != 1:
            print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
            exit()
        # Append the face encoding found in that image to the list of face encodings we have
        face_encodings.append(face_encodings_in_image[0].tolist())
        now = dt.datetime.now(tz=tz)
        path_to_cloud="Known_faces/"+phone_number+"_"+str(now.timestamp())+".jpg"
        #to upload the image in storage
        public_url=storage.child(path_to_cloud).put(path_to_image)
        public_url=storage.child(path_to_cloud).get_url(None)
        urls.append(public_url)
    db = firebase.database()
    now = dt.datetime.now(tz=tz)
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    data_to_upload = {
        "name":name,
        "url": urls,
        "encoding": face_encodings,
        "status": status,
        "RegisteredBy": sec_code,   # "SEC001",
        "RegisteredOn": dt_string,
        "Contact No": phone_number,
        "ExpiryDate": expdate,
        "Designation": designation
    }
    add='RegisteredPerson/'+phone_number+'/'
    db.child(add).set(data_to_upload)
    # print("Uploaded"+path_to_image)


print('Done')