import cv2
import numpy as np
import matplotlib.pyplot as plt

grey = cv2.imread('c:/img/House256rgb.png', 0)
cv2.imshow('original grey', grey)
cv2.waitKey()

hist, bins = np.histogram(grey, 256, [0, 255])
plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

grey_eq = cv2.equalizeHist(grey)
hist, bins = np.histogram(grey_eq, 256, [0, 255])
plt.fill_between(range(256), hist, 0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow('equallized grey', grey_eq)
cv2.waitKey()

