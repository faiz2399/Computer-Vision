import cv2 as cv
import numpy as np
import pyautogui
##from matplotlib import pyplot as plt


cam = cv.VideoCapture(0)

##lower = np.array([20,70,70])
##upper = np.array([80,255,255])

lower = np.array([120,100,100])
upper = np.array([40,255,255])


lower_g = np.array([50,100,100])
upper_g = np.array([80,255,255])

while(cam.isOpened()):
    ret,frame = cam.read()
    frame = cv.flip(frame,1)
    img_smooth = cv.GaussianBlur(frame,(7,7),0)
    mask = np.zeros_like(frame)
    mask[50:350 , 50:350] = [255,255,255]
    image_roi = cv.bitwise_and(img_smooth,mask)

    cv.rectangle(frame,(50,50),(350,350),(0,0,255),2)
    cv.line(frame,(150,50),(150,350),(0,0,255),1)
    cv.line(frame,(250,50),(250,350),(0,0,255),1)
    cv.line(frame,(50,150),(350,150),(0,0,255),1)
    cv.line(frame,(50,250),(350,250),(0,0,255),1)

    
    img_hsv = cv.cvtColor(image_roi,cv.COLOR_BGR2HSV)
    img_threshold = cv.inRange(img_hsv,lower,upper)

    contour,heirachy = cv.findContours(img_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

    if(len(contour)!=0):
        areas = [cv.contourArea(c) for c in contour]
        max_index = np.argmax(areas)
        cnt = contour[max_index]

        M = cv.moments(cnt)

        if(M['m00']!=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv.circle(frame,(cx,cy),4,(0,255,0),-1)

            if cx<150:
                dist_x = -20
            elif cx > 250:
                dist_x = 20
            else:
                dist_x = 0

            if cy<150:
                dist_y = -20
            elif cy > 250:
                dist_y = 20
            else:
                dist_y = 0

            pyautogui.moveRel(dist_x,dist_y,duration = .25)

        img_threshold_green = cv.inRange(img_hsv,lower_g,upper_g)

        contour_g,heirachy = cv.findContours(img_threshold_green,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        if(len(contour_g)!=0):
               pyautogui.click()
               cv.waitKey(1000)
        
                    
                
   
    
    cv.imshow('frame',frame)

    key = cv.waitKey(20)
    if key == ord('s') or key==27:
        break

##img_RGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
##plt.imshow(img_RGB)
##plt.show()

cam.release()
cv.destroyAllWindows()
