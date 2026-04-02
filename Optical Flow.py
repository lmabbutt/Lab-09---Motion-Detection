import numpy as np
import cv2
import time

threshold = 30
step = 12  # the number of pixels within each block the velocity is being looked for

def draw_flow(img, flow):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)
    return img_bgr

###################################
# Parameters for dense optical flow using Gunnar Farneback's algorithm.

# Computed flow image that has the same size as prev and type CV_32FC2
flow = None

# Image scale (<1) to build pyramids; 0.5 = classical pyramid (each layer half the previous)
pyrScale = 0.5

# Number of pyramid layers including initial image; 1 = only original images used
levels = 3

# Averaging window size; larger = more robust to noise but more blurred motion field
winsize = 15

# Number of iterations at each pyramid level
iterations = 3

# Pixel neighborhood size for polynomial expansion; typically 5 or 7
polyN = 5

# Standard deviation of Gaussian used to smooth derivatives;
# for polyN=5 use polySigma=1.1, for polyN=7 use polySigma=1.5
polySigma = 1.1

# Operation flags: combination of OPTFLOW_USE_INITIAL_FLOW and/or OPTFLOW_FARNEBACK_GAUSSIAN
flags = 0

cap = cv2.VideoCapture(1)
suc, prev = cap.read()

# ---- Lab Step c-i: Comment out this section to switch to grayscale tracking ----
## Separate into color layers to get Red Only
#red = np.matrix(prev[:,:,2])
#blue = np.matrix(prev[:,:,0])
#green = np.matrix(prev[:,:,1])
#red_Only = np.int16(red) - np.int16(green) - np.int16(blue)

# Threshold to B/W image
#red_Only[red_Only < threshold] = 0
#red_Only[red_Only >= threshold] = 255

# Converting to correct image type values
#red_Only = np.uint8(red_Only)

# Save this as the previous red only image
#prevRed = red_Only
# ---- End of section to comment out for grayscale ----

# ---- Lab Step c-ii: Uncomment the line below to switch to grayscale tracking ----
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    suc, img = cap.read()

    # ---- Lab Step c-iii: Comment out this section to switch to grayscale tracking ----
    ## Separate into color layers to get Red Only
    #red = np.matrix(img[:,:,2])
    #blue = np.matrix(img[:,:,0])
    #green = np.matrix(img[:,:,1])
    #red_Only = np.int16(red) - np.int16(green) - np.int16(blue)

    # Threshold to B/W image
    #red_Only[red_Only < threshold] = 0
    #red_Only[red_Only >= threshold] = 255

    # Converting to correct image type values
    #red_Only = np.uint8(red_Only)
    # ---- End of section to comment out for grayscale ----

    # Take current image and convert to grayscale for better visual later
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Start time to calculate FPS
    start = time.time()

    # ---- Lab Step c-iv: Comment out these two lines to switch to grayscale tracking ----
    # Computes dense optical flow using Gunnar Farneback's algorithm (Red Only)
    #flow = cv2.calcOpticalFlowFarneback(prevRed, red_Only, flow, pyrScale, levels, winsize, iterations, polyN, polySigma, flags)
    #prevRed = red_Only
    # ---- End of section to comment out for grayscale ----

    # ---- Lab Step c-v: Uncomment these two lines to switch to grayscale tracking ----
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, flow, pyrScale, levels, winsize, iterations, polyN, polySigma, flags)
    prevgray = gray

    # End time
    end = time.time()

    # Calculate the FPS for current frame detection
    fps = 1 / (end - start)
    print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))

    key = cv2.waitKey(5)    
    if key == 27:  # esc key to break out of the while loop
        break

cap.release()
cv2.destroyAllWindows()
