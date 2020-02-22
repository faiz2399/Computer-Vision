import cv2 as cv
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor1.dat")

(mstart,mend) = (48,67)

smile_const = 5

counter = 0
selfie_no = 0

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return(x,y,w,h)

def shape_to_np(shape, dtype ="int"):
    coords = np.zeros((68,2),dtype=dtype)

    for i in range(0,68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        
    return coords

def smile(shape):

    left = shape[48]
    right = shape[54]

    mid = (shape[51] + shape[62] + shape[66] + shape[57])/4

    dist = np.abs(np.cross(right - left,left - mid))/np.linalg.norm(right - left)

    return dist

cam = cv.VideoCapture(0)

while(cam.isOpened()):

    

    ret,image = cam.read()
    image = cv.flip(image,1)

    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)




    rects = detector(gray,2)

    for i in range(0,len(rects)):

        (x,y,w,h) = rect_to_bb(rects[i])

        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(image,"Face : {}".format(i+1),(x-10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        shape = predictor(gray,rects[i])
        shape = shape_to_np(shape)

        mouth = shape[mstart:]
        for(x,y) in mouth:
            cv.circle(image,(x,y),1,(255,255,255),-1)

        smile_param = smile(shape)
        cv.putText(image, "SP: {:.2f}".format(smile_param),(300,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        if smile_param > smile_const:
            cv.putText(image, "Smile detected",(300,60),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            counter +=1
            if counter >=15:
                selfie_no+=1
                ret,frame = cam.read()
                img_name = "selfie.png".format(selfie_no)
                cv.imwrite(img_name,frame)

                print("{} taken!".format(img_name))
                counter = 0

        else:
            counter = 0
            

    cv.imshow("live_face", image)
    key = cv.waitKey(25)

    if key == 27:
        break
         

cam.release()
cv.destroyAllWindows()

    
