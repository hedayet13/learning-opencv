import numpy as np
import cv2 as cv
import numpy as np
import pylab as plt
image = cv.imread('retina2.jpg')

resized= cv.resize(image,(640,440))
gray= cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
sobelx =cv.Sobel(gray,cv.CV_64F,1 ,0,ksize=5)
sobely=cv.Sobel(gray,cv.CV_64F,0,1,ksize=5)
laplacian = cv.Laplacian(gray,cv.CV_64F)
blended = cv.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=.5,gamma=0)
ret, th1 = cv.threshold(gray,100,255,cv.THRESH_BINARY)
shap = gray.shape
calc= cv.calcHist(th1,channels=[0],mask=None,histSize=[256],ranges=[0,256])
# plt.plot(calc)
# plt.show()
equ_hist= cv.equalizeHist(gray)

print(shap)
equHist = cv.equalizeHist(gray)
calc= cv.calcHist(equ_hist,channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.plot(calc)
plt.show()
cv.imshow('resized',equ_hist)
# cv.imshow('resized Image',th1)
cv.waitKey(0)