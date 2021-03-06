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
#import firebase
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto import Random


Config = {
    "apiKey": "AIzaSyBqH-1SFbLkCjaW_Qilj7TQq-2E3cy43FQ",
    "authDomain": "iot-project-e0803.firebaseapp.com",
    "databaseURL": "https://iot-project-e0803-default-rtdb.firebaseio.com/",
    "projectId": "iot-project-e0803",
    "storageBucket": "iot-project-e0803.appspot.com",
    "messagingSenderId": "292898203599",
    "appId": "1:292898203599:web:382bc1a3eb5c9e28c4d4ba",
    "measurementId": "G-FY4R61HR0P"
  };

# First let us encrypt secret message
BLOCK_SIZE = 16
pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * chr(BLOCK_SIZE - len(s) % BLOCK_SIZE)
unpad = lambda s: s[:-ord(s[len(s) - 1:])]
 
password = "0A1B2C3D4E5F6A7B8C9D0E1F2A3B3C4D"
 
 
def encrypt(raw):
    res = []
    for i in raw:
        i = str(i)    
        private_key = hashlib.sha256(password.encode("utf-8")).digest()
        i = pad(i).encode()
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(private_key, AES.MODE_CBC, iv)
        res.append(base64.b64encode(iv + cipher.encrypt(i)).decode())
    return res

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
    print(paths_to_images)
    for path_to_image in paths_to_images:
        # Get face encodings from the image
        face_encodings_in_image = get_face_encodings(path_to_image)
        print(len(face_encodings_in_image))
        # Make sure there's exactly one face in the image
        if len(face_encodings_in_image) != 1:
            print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
            exit()
        # Append the face encoding found in that image to the list of face encodings we have
        face_encodings.append(encrypt(face_encodings_in_image[0]))
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

# CALL UPLOAD() FUNTION HERE

print('Done')