import cv2
import sys
import time

# Get user supplied values
# imagePath = sys.argv[0]
cascPath = "path of xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image, the pixel should be 640*480
path = "path of jpg or png"
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
