import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

nadia = cv.imread('pic/nadia.jpg',0)
denis =cv.imread('pic/denis.jpg',0)
solvey = cv.imread('pic/solvey.jpg',0)
face_cascade= cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
def detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img)
    for(x,y,w,h) in face_rect:
        cv.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img
result = detect_face(nadia)
# cv.imshow('dd',result)
# plt.imshow(detect_face(nadia),cmap='gray')
# plt.imshow(detect_face(denis),cmap='gray')
# plt.imshow(detect_face(solvey),cmap='gray')
# cv.imshow('solvey',denis)
def adj_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
    for(x,y,w,h) in face_rect:
        cv.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    return face_img

# plt.imshow(adj_detect_face(solvey),cmap='gray')

eye_cascade =cv.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
def detect_eyes(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img,scaleFactor=5,minNeighbors=5)
    for(x,y,w,h) in eye_rect:
        cv.rectangle(eye_img,(x,y),(x+w,y+h),(255,255,255),10)
    return eye_img
# plt.imshow(detect_eyes(denis),cmap='gray')
# plt.imshow(solvey,cmap='gray')

def detect_and_blur_plates(img):
    plate_img = img.copy()
    roi = img.copy()
    plate_rects = face_cascade.detectMultiScale(plate_img,scaleFactor=1.3,minNeighbors=3)
    for (x,y,w,h) in plate_rects:
        roi = roi[y:y+h,x:x+w]
        blurred_roi  = cv.medianBlur(roi,25)
        plate_img[y:y+h,x:x+h]=blurred_roi
    return plate_img
# result= detect_and_blur_plates(nadia)
# cv.imshow('re',result)
cap =cv.VideoCapture(0)

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    cv.imshow('Video face detection',frame)
    k = cv.waitKey(1)
    if k == 27 :
        break
cap.release()
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()