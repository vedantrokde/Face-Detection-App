# importing libraries
import cv2

# creating instance for video capturing
video = cv2.VideoCapture(0)

# loading face haarcascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    # reading video stream
    check, frame = video.read()

    # convert image to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting faces using haarcascade
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

    # marking faces on image
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #  display the frame
    cv2.imshow("Capturing", frame)
    
    # checking input key
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

# releasing video instance
video.release()
cv2.destroyAllWindows()