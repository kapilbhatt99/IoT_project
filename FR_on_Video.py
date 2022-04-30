from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from google.colab.patches import cv2_imshow
import dlib
import scipy.misc
import numpy as np
import cupy as cp
import os
import imageio
import cv2
import time
from datetime import datetime as dt
import pytz

weights = '/content/drive/My Drive/ML project/MODELS/mmod_human_face_detector.dat'
# weights = '/content/drive/My Drive/ML project/mmod_human_face_detector (1)_1.dat'

# initializing cnn face detector model
# This allows us to detect faces in images
cnn_face_detector = dlib.cnn_face_detection_model_v1(weights)
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
TOLERANCE = 0.50

def rect_to_bb(face):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y
    return (x, y, w, h)

# This function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    # Load image
    image = cv2.imread(path_to_image)
    # Detect faces using the face detector
    res = cnn_face_detector(image,1)
    shapes_faces = []
    for r in res:
        face = r.rect
        # Get pose/landmarks of those faces
        # Will be used as an input to the function that computes face encodings
        # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
        shapes_faces.append(shape_predictor(image,face))
        
    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def get_vid_encodings(v_image,face):
    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
    shapes_faces = shape_predictor(v_image, face)
    # For every face detected, compute the face encodings
    # return [np.array(face_recognition_model.compute_face_descriptor(v_image, face_pose, 1)) for face_pose in shapes_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(v_image, shapes_faces, 1))]

# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # Return an array with True/Face values based on whether or not a known face matched with the given face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the TOLERANCE value
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)
 
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
 
    # Top left
    img = cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    img = cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    img = cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right
    img = cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    img = cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    img = cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left
    img = cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    img = cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    img = cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right
    img = cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    img = cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    img = cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    return img

# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings(known_faces, face)
    # Return the name of the first match
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    # Return not found if no match found
    return 'Not Found'

# Get path to all the known images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('/content/drive/My Drive/ML project/known faces/'))
# Sort in alphabetical order
image_filenames = sorted(image_filenames)
# Get full paths to images
paths_to_images = ['/content/drive/My Drive/ML project/known faces/' + x for x in image_filenames]
# List of face encodings we have
face_encodings = []
# Loop over images to get the encoding one by one
start = time.time()
for path_to_image in paths_to_images:
    # Get face encodings from the image
    face_encodings_in_image = get_face_encodings(path_to_image)
    # Make sure there's exactly one face in the image
    if len(face_encodings_in_image) != 1:
        print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
        exit()
    # Append the face encoding found in that image to the list of face encodings we have
    face_encodings.append(face_encodings_in_image[0])
end = time.time()
print("Execution Time for Encoding: {}".format(end-start))

# Get list of names of people by eliminating the .JPG OR .PNG extension from image filenames
names = [x[:-4] for x in image_filenames]

count = 0
suc=True
cap = cv2.VideoCapture('/content/drive/My Drive/ML project/test/test.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
#out = cv2.VideoWriter('outpy.avi',fourcc, 20, (frame_width,frame_height))

# set time zone
tz = pytz.timezone("Asia/Calcutta")
print("Video Capture Started")
start = time.time()
skip = int(FPS);

skip = 0;
while suc:
    #Read the frame
    for t in range(skip):
        cap.grab()
    suc, img = cap.read()

    if suc == False:
        print('False')
        break
    else:
        # rotate stream by 270 deg
        img = np.rot90(img,3,(0,1))
        imgframe = img.copy()
        # detect faces in the image
        result_list = cnn_face_detector(img, 1)
        # display faces on the original image
        # plot each face as a subplot
        org = (30, 30)
        # font 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # fontScale 
        fontScale = 0.6
        # Blue color in BGR 
        color = (255, 255, 255) 
        # Line thickness of 2 px 
        thickness = 1
        
        # get current date and time
        now = dt.now(tz=tz) 
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        d_string = now.strftime("%d/%m/%Y")
        t_string = now.strftime("%H:%M:%S")
                    
        framestamp = 'Frame{}'.format(count) + ' ' + dt_string

        # Using cv2.putText() method 
        imgframe = cv2.putText(imgframe, framestamp, org, font,fontScale, color, thickness, cv2.LINE_AA)
        for i in range(len(result_list)):
            # get coordinates
            x1, y1, width, height = rect_to_bb(result_list[i])
            x2, y2 = x1 + width, y1 + height
            image = img[y1-40:y2+40, x1-40:x2+40]
            res = result_list[i].rect
            face_encodings_in_image = get_vid_encodings(img,res)
            if len(face_encodings_in_image) != 1:
                print("Please change image: - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
                # continue
            else:
                # Find match for the face encoding found in this test image
                match = find_match(face_encodings, names, face_encodings_in_image[0])
                # match = find_match(face_encodings, names, face_encodings_in_image[0])
                    # Print the path of test image and the corresponding match
                    # " + path_to_image + "
                filename = "/content/drive/My Drive/ML project/images/" + match + "_{}".format(count) +"_{}.jpg".format(i)
                #filename = "" + match + "_{}".format(count) +"_{}.jpg".format(i)
                if image.any():
                    # font 
                    font = cv2.FONT_HERSHEY_SIMPLEX 
                    # fontScale 
                    fontScale = 0.25
                    # Blue color in BGR 
                    color = (255, 255, 255) 
                    # Line thickness of 2 px 
                    thickness = 1
                    # Using cv2.putText() method 
                    # image = cv2.putText(image, d_string, (5,10), font,fontScale, color, thickness, cv2.LINE_AA)
                    # image = cv2.putText(image, t_string, (5,20), font,fontScale, color, thickness, cv2.LINE_AA)
                    cv2.imwrite(filename,image)
                print("{} : This is".format(count),match)
                if match != "Not Found":
                    ### Drawing Border around the detected face
                    # color is in ( B, G, R) format
                    imgframe = draw_border(imgframe, (x1, y1), (x2, y2), (0, 255, 0),2, 5, 5).copy()
                    ###
                else:
                    ### Drawing Border around the detected face
                    # color is in ( B, G, R) format
                    imgframe = draw_border(imgframe, (x1, y1), (x2, y2), (0, 0, 255),2, 5, 5).copy()
                    ###
        cv2.imwrite("/content/drive/My Drive/ML project/images/Frame{}.jpg".format(count),imgframe)
        imgframe = cv2.cvtColor(imgframe,cv2.COLOR_RGB2BGR)
        #out.write(imgframe)
        count = count + 1

end = time.time()
print("Recognition time: {}".format(end - start))
print("Total Frames: ",frames)
print("FPS: ",FPS)
print("Skip: ",skip)
print("Total Frames Processed: ",count)
cap.release()
#out.release()
cv2.destroyAllWindows()
print('Done')