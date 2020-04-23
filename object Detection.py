
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

full =cv.imread('pic/retina2.jpg')

# shi-tomasi corner detection
flat_chess_board = cv.imread('pic/flatChessBoard.jpg')
flat_chess_board= cv.resize(flat_chess_board,(640,440))
flat_chess_gray= cv.cvtColor(flat_chess_board,cv.COLOR_BGR2GRAY)
full = cv.cvtColor(full,cv.COLOR_BGR2RGB)
# corners= cv.goodFeaturesToTrack(flat_chess_gray,100,0.01,10)
# corners=np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv.circle(flat_chess_board,(x,y),3,(0,255,255),-1)
# cv.imshow('flatChess',flat_chess_board)
# cv.imshow('full',full)

# canny-edge detection
# blur the image first to detect the image correctly , agian try to get the high median value of the image .
# More the median value of the image ,more it helps to get the edge. .. med_val=np.median(image)...
# increase the threshold value or find out the fixed threshold value or get the minimum and maximum threshold value
img = cv.imread('pic/sammy.jpg')
edges=cv.Canny(img,threshold1=127,threshold2=127)


# cv.imshow('edge',edges)
# print(np.median(img))
# print(img.shape)

# Grid detection(use flat chess board image)
found,corners = cv.findChessboardCorners(flat_chess_board,(9,6))
cv.drawChessboardCorners(flat_chess_board,(9,6),corners,found)
# print(found)
# dots=cv.imread('circle.png')
# _,corners=cv.findCirclesGrid(dots,(10,10),cv.CALIB_CB_SYMMETRIC_GRID)
# cv.drawChessboardCorners(dots,(10,10),corners)
# cv.imshow('img',flat_chess_board)


#contourDetection
img = cv.imread('pic/contours.png')
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
contours,hierarchy= cv.findContours(img,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
external_contours = np.zeros(img.shape)
for i in range (len(contours)):
    # externalcontour
    if hierarchy[0][i][3]==-1:
        cv.drawContours(external_contours,contours,i,255,-1)
internal_contours = np.zeros(img.shape)
for i in range (len(contours)):
    # internalcontour
    if hierarchy[0][i][3]!=-1:
        cv.drawContours(internal_contours,contours,i,255,-1)


# print(hierarchy)
# print(len(contours))
# cv.imshow('contours',internal_contours)
# cv.imshow('external',external_contours)


cv.waitKey(0)
