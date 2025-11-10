import numpy as np
import cv2 as cv

cap = cv.VideoCapture(r'resources\raw_data\caroline_shot\stream00.avi')
fgbg = cv.createBackgroundSubtractorMOG2()
idx = 0
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    idx+=1

    if idx == 200:
        cv.imshow("preview", fgmask)
        cv.waitKey(0)
    
cap.release()
cv.destroyAllWindows()