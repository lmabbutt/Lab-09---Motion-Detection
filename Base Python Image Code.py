import numpy as np
import cv2

cap=cv2.VideoCapture(1)# number is for the camera 0 & up

while(1):

    _,frame = cap.read()

    red = np.matrix(frame[:,:,2])
    blue = np.matrix(frame[:,:,0])
    green = np.matrix(frame[:,:,1])

    red_Only = np.int16(red)-np.int16(green)-np.int16(blue)

    red_Only[red_Only<0] = 0
    red_Only[red_Only>255] = 255

    column_sums = np.matrix(np.sum(red_Only,0))
    column_numbers = np.matrix(np.arange(640))
    column_mult = np.multiply(column_sums,column_numbers)
    total = np.sum(column_mult)
    total_total = np.sum(np.sum(red_Only))
    column_location = total/total_total

    print(column_location)

    red_Only=np.uint8(red_Only)

    cv2.imshow('rgb',frame)
    cv2.imshow('Red Only',red_Only)

    k=cv2.waitKey(5)
    if k==27: #esc key to break out of the while loop
        break

cv2.destroyAllWindows()