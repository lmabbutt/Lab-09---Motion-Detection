# Name
# Code description

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(1)

while(1):
    # First Image
    _, frame = cap.read()

    red1 = np.matrix(frame[:,:,2])
    blue1 = np.matrix(frame[:,:,0])
    green1 = np.matrix(frame[:,:,1])

    red_Only1 = np.int16(red1) - np.int16(green1) - np.int16(blue1)

    red_Only1[red_Only1<0] = 0
    red_Only1[red_Only1>255] = 255

    red_Only1 = np.uint8(red_Only1)

    time.sleep(0.01) #Sleep time in sec

    # Second Image
    _, frame = cap.read()

    red2 = np.matrix(frame[:,:,2])
    blue2 = np.matrix(frame[:,:,0])
    green2 = np.matrix(frame[:,:,1])

    red_Only2 = np.int16(red2) - np.int16(green2) - np.int16(blue2)

    red_Only2[red_Only2<0] = 0
    red_Only2[red_Only2>255] = 255

    red_Only2 = np.uint8(red_Only2)

    # Find the motion of the red object
    Motion = red_Only2 - red_Only1
    Motion[Motion<0] = 0
    Motion[Motion>255] = 255

    # Find the column location of the motion of the red object
    column_sums = np.matrix(np.sum(Motion, 0))
    column_numbers = np.matrix(np.arange(640))
    column_mult = np.multiply(column_sums, column_numbers)
    total = np.sum(column_mult)
    total_total = np.sum(np.sum(Motion))
    column_location = total / total_total

    print(column_location)

    Motion = np.uint8(Motion)

    cv2.imshow('rgb', frame)
    cv2.imshow('Motion', Motion)

    k = cv2.waitKey(5)
    if k == 27:  # esc key to break out of the while loop
        break

cv2.destroyAllWindows()
