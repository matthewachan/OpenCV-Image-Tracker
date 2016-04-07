import cv2
import numpy as np
from matplotlib import pyplot as plt

""" LOAD THE IMAGES / VIDEOS """
# load the video
video = cv2.VideoCapture('RyanRun.MP4')
# load the image to track in the video
hipImg = cv2.imread('hip.png', 0)
# get the width and height of the image
wdt, hgt = hipImg.shape[::-1]


""" SETUP VARIABLES """
# image comparison methods
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# keep track of the current frame of the video playing
currFrame = 0

# store x and y coordinates of the hip in the video
xCoords = list()
yCoords = list()


""" DISPLAY THE VIDEO """
while (video.isOpened()):
    
    currFrame += 1     
    
    # skip the unnecessary frames
    if 0 <= currFrame < 880:
        video.grab()
        
    elif 880 <= currFrame < 1072:
        # capture the current frame
        ret, frame = video.read()
        # slice the array and get rid of unneeded data
        frame = frame[:,:,0]
        # convert the current frame the grayscale
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # compare the hip image to the current frame
        compareMethod = eval(methods[5])
        res = cv2.matchTemplate(frame, hipImg, compareMethod)
        # copy the frame to avoid data corruption (strange opencv wrapper bug)    
        frame = frame.copy()    
        
        # grab the dimensions of the resulting image from the comparison
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        # store x and y values of the hip 
        xCoords.append(minLoc[0])
        yCoords.append(minLoc[1])
        topLeft = minLoc
        bottomRight = (topLeft[0] + wdt, topLeft[1] + hgt)
        cv2.rectangle(frame, topLeft, bottomRight, 255, 2)
        # show the image on the screen
        cv2.imshow('frame', frame)
        # exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        break
    
    
""" CLOSE THE VIDEO """
# close the video / window
video.release()
cv2.destroyAllWindows()


""" PLOT THE POSITION OF THE HIP THROUGHOUT TIME """
# flip the y values
yCoords = [-y for y in yCoords]
plt.plot(xCoords, yCoords)
plt.axis([0, 600,-120, -50])