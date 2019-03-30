#rice
import cv2
import numpy as np

fixed_size = tuple((500, 500))

im = cv2.imread("../input/set123/rice.jpg")
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
l_green = np.array([65,60,60])
u_green = np.array([80,255,255])
mask = cv2.inRange(hsv, l_green, u_green)

import matplotlib.pyplot as plt

#plt.imshow(mask)

res = cv2.bitwise_and(hsv,hsv, mask=mask)
# #import matplotlib.pyplot as plt

plt.imshow(res)

_,threshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
#plt.imshow(threshed,cmap='gray')
threshed = cv2.dilate(threshed)

closing = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

# contours,hier = cv2.findContours(threshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(closing,cmap='gray')

#training
#1->resize
#2->normalize

#im_path=""
ima = cv2.bitwise_and(im,threshed,mask = mask )
plt.imshow(ima)

contours,hierarchy = cv2.findContours(ima, 1, 2)
plt.imshow(contours)

area = cv2.contourArea(contours)
print(area)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split


