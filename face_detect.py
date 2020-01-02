import cv2
import sys
import time

# Get user supplied values
# imagePath = sys.argv[0]
cascPath = "/home/fiftycentsjj/Downloads/raspberry_beginner_lecture/robot_race/FaceDetect-master/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
path = "/home/fiftycentsjj/Downloads/raspberry_beginner_lecture/robot_race/little_mix_right.jpg"
image = cv2.imread(path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # read the image with hsv
cv2.imshow("image", image)
# time.sleep(3)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# color = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
# Detect faces in the image
faces = faceCascade.detectMultiScale(
    hsv_image,
    scaleFactor=1.0358, # object confidence
    minNeighbors=5,
    minSize=(5, 5),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# time.sleep(3)
cv2.imshow("Faces found", image)
cv2.waitKey(0)
