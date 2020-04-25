
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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

# feature matching(having problem in orb method)***********Basics of Brute-Force Matcher
# follow the link 'https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html'
reeses = cv.imread('pic/reeses.jpg',0)
# reeses = cv.resize(reeses,(400,500))
# reeses = cv.cvtColor(reeses,cv.COLOR_BGR2GRAY)
cereals = cv.imread('pic/cereals.jpg',0)
# cereals = cv.resize(cereals,(500,400))
# cereals= cv.cvtColor(cereals,cv.COLOR_BGR2GRAY)
# print(cereals.shape)
# orb = cv.ORB_create()
# kp1 , des1 = orb.detectAndCompute(reeses,None)
# kp2 , des2 = orb.detectAndComupte(cereals,None)
# bf =cv.BFMatcher(cv.NORM_HAMMING,crosscheck =True)
# matches = bf.match(des1,des2)
# matches =sorted(matches, key=lambda x:x.distance)
# reeses_matches = cv.drawMatches(reeses,kp1,cereals,kp2,matches[:25],outImg=None,flags=2)
#
# plt.imshow(reeses_matches)
# plt.show()
# cv.imshow('sd',reeses_matches)
# cv.imshow('ads',matches)
#
# cv.imshow('reeses',cereals)

# sift =cv.SIFT()
# sift = cv.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params=dict(algorithm =FLANN_INDEX_KDTREE,trees= 5)
search_params = dict(checks =50)
flann =cv.FlannBasedMatcher(index_params,search_params)





# watershad techniques(median,grayscale , binary threshold,find contours)

sep_coin = cv.imread('pic/coin.jpg')
sep_blur = cv.medianBlur(sep_coin,5)
gray_sep_coins = cv.cvtColor(sep_blur,cv.COLOR_BGR2GRAY)
ret, sep_thresh = cv.threshold(gray_sep_coins,160,255,cv.THRESH_BINARY_INV)
contours, hierarchy = cv.findContours(sep_thresh.copy(), cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

for i in range (len(contours)):
    if hierarchy[0][i][3]==-1:
        cv.drawContours(sep_coin,contours,i,(255,0,0),2)
img = cv.imread('pic/coin1.jpg')
img =cv.medianBlur(img,1)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
kernel = np.ones((3,3),np.uint8)
opening =cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel)
sure_bg=cv.dilate(opening,kernel,iterations=1)
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret,sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
ret ,markers = cv.connectedComponents(sure_fg)
# print(markers)
markers=markers+1
markers[unknown==255] = 0
markers =cv.watershed(img,markers)
contours, hierarchy = cv.findContours(markers.copy(), cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

for i in range (len(contours)):
    if hierarchy[0][i][3]==-1:
        cv.drawContours(sep_coin,contours,i,(255,0,0),2)


road  = cv.imread("pic/road.jpg")
road_Copy = np.copy(road)
# cv.imshow('road',road_Copy)
# print(road_Copy.shape)
marker_image = np.zeros(road.shape[:2],dtype=np.int32)
segments =np.zeros(road.shape,dtype=np.uint8)

def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)
colors = []
for i in range(10):
    colors.append(create_rgb(i))
# print(colors)

# global variable
n_markers = 10  #  0-9
current_marker = 1
marks_updates =False

# callback function
def mouse_talkback(event,x,y,flags,param):
    global marks_updates
    if event==cv.EVENT_LBUTTONDOWN:
        # markers passed to the watershed algorithm
        cv.circle(marker_image,(x,y),10,(current_marker),-1)
        cv.circle(road_Copy,(x,y),10,colors[current_marker],-1)
        marks_updates=True


# while true
cv.namedWindow('Road Image ')
cv.setMouseCallback('Road Image ',mouse_talkback)

while True:
    cv.imshow('Watershed segments ', segments)
    cv.imshow('Road Image',road_Copy)

    # cloase all the widwos
    k = cv.waitKey(1)
    if k==27:
        break

    # clearing all the colors pressing C  key
    elif k==ord('c'):
        road_Copy=road.copy()
        marker_image = np.zeros(road.shape[:2],dtype=np.int32)
        segments = np.zeros(road.shape,dtype= np.uint8)


    # update color choice
    elif k>0 and chr(k).isdigit():
        current_marker=int(chr(k))

    if marks_updates:
        marker_image_copy = marker_image.copy()
        cv.watershed(road,marker_image_copy)
        segments=np.zeros(road.shape,dtype=np.uint8)

        for color_ind in range(n_markers):
            # coloring segment using numpy call
            segments[marker_image_copy==(color_ind)]= colors[color_ind]


# plt.imshow(road)
# cv.imshow('disty',sep_coin)
cv.waitKey(0)
cv.destroyAllWindows()
