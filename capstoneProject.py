import cv2 as cv
import numpy as np

from sklearn.metrics import pairwise

background = None

accumulated_weight = 0.5

roi_top= 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calc_acccum_avg(frame ,accumulated_weight):

    global background

    if background is None:
        background = frame.copy().astype('float')
        return None
    cv.accumulateWeighted(frame , background , accumulated_weight)

def segment (frame ,thresholdMin = 25):

    diff = cv.absdiff(background.astype('uint8'),frame)
    ret,thresholded = cv.threshold(diff,thresholdMin,255,cv.THRESH_BINARY)
    image ,contours, hierarchy = cv.findContours(thresholded.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    else:
        hand_segment = max(contours,key=cv.contourArea)

        return (thresholded,hand_segment)

def count_fingers(thresholded ,hand_segment):
    conv_hull = cv.convexHull(hand_segment)

    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    left = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    right= tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])

    cX = (left[0]+right[0])//2
    cY = (top[1]+bottom[1])//2

    distance =pairwise.euclidean_distances([cX,cY],Y= [left,right,top,bottom])[0]

    max_distance = distance.max()

    radius = int(.9*max_distance)
    circuference = (2*np.pi*radius)


    circular_roi = np.zeros(thresholded[:2],dtype='uint8')


    cv.circle(circular_roi,(cX,cY),radius,255,10)

    circular_roi = cv.bitwise_and(thresholded,thresholded,mask=circular_roi)

    image ,contours ,hierarchy = cv.findContours(circular_roi.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

    count = 0
    for cnt in contours :
        (x,y,w,h) =cv.boundingRect(cnt)
        out_of_wrist = (cY+(cY*.25))>(y+h)

        limit_points =((circuference*.25)>cnt.shape[0])

        if out_of_wrist and limit_points:
            count +=1

    return count

cam = cv.VideoCapture(0)

num_frames = 0
while True:



    ret,frame =cam.read()
    frame_copy = frame.copy()
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    gray =cv.cvtColor(roi,cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray , (7,7),0)
    if num_frames <60 :
        calc_acccum_avg(gray,accumulated_weight)
        if num_frames<=59:
            cv.putText(frame_copy,"WAit . Getting background",(200,200),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv.imshow('Finger count ',frame_copy)
    else:
        hand  = segment(gray)

        if hand is not  None :
            thresholed , hand_segment  = hand
            cv.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            fingers  = count_fingers(thresholed,hand_segment)
            cv.putText(frame_copy,str(fingers),(70,50),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            cv.imshow('Thresholded ',thresholed)

    cv.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),6)
    num_frames +=1

    cv.imshow('finger count ',frame_copy)

    k= cv.waitKey(1) & 0xFF

    if k ==27 :
        break

cam.release()
cv.destroyAllWindows()