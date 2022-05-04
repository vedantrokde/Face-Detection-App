# importing libraries
import cv2

# loading face haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# loading the image
# img = cv2.imread("news.jpg")
img = cv2.imread("photo.jpg")

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detecting faces using haarcascade
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# marking faces on image
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

# resizing the image
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

# showing image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()