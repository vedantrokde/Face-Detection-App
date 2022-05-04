# importing libraries
import cv2
import pandas as pd
from datetime import datetime

# creating instance for video capturing
video = cv2.VideoCapture(0)

# declaring variables
first_frame = None
status_list = [0, 0]
times = []

while True:
    status = 0
    # reading video stream
    check, frame = video.read()

    # convert image to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_img
        continue

    # detecting motion using first frame
    delta_frame = cv2.absdiff(first_frame, gray_img)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)

    # finding countors from movement
    cnts, _ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # plotting the counter markers
    for countour in cnts:
        if cv2.contourArea(countour) < 1000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(countour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # motion record
    status_list.append(status)
    status_list.pop(0)
    
    if status_list[0]^status_list[1]:
        times.append(datetime.now())

    # display the frame
    cv2.imshow("Motion Detector", frame)
    
    # checking input key
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

# checking for inconsistent data
if len(times)%2!=0:
    times.append(datetime.now())

# releasing video instance
video.release()
cv2.destroyAllWindows()

# storing motion recording data as csv
if len(times)>=2:
    pd.DataFrame(data={'Start': times[::2], 'End': times[1::2]}, columns=['Start', 'End']).to_csv("motion_record.csv")