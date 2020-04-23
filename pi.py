import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

blank_img= np.zeros(shape=(512,512,3),dtype=np.uint8)
print(blank_img.shape)

cv.rectangle(blank_img,pt1=(150,400),pt2=(400,150),color=(0,255,0),thickness=10)
plt.imshow(blank_img)
plt.show()

cv.waitKey(0)