from this import d
from turtle import width
import cv2
import dlib


# read the image
img = cv2.imread("avatar3.jpg")


# convert img to grayscale:3D -> 2D
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# dlib: Load Face Recongnition Detector

face_detector = dlib.get_frontal_face_detector()

# load the predictor

predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces = face_detector(gray)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # Draw rectangle face
    cv2.rectangle(img=img, pt1=(x1, y1), pt2=(
        x2, y2), color=(0, 255, 0), thickness=3)
    
    face_features = predictor(image=gray,box=face)

    # Loop through all 68 points

    for n in range(0, 68):
      x=face_features.part(n).x
      y=face_features.part(n).y

      # Draw a circle 
      cv2.circle(img=img, center=(x, y), radius=5, color=(255, 0, 0), thickness=3)


# show the image

cv2.imshow(winname="Face Recongnition App", mat=img)

# wait for a key press to exit

cv2.waitKey(delay=0)

# CLose All windows

cv2.destroyAllWindows()
