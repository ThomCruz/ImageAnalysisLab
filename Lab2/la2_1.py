import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("lenna.png",0) 
row, col = img.shape[:2]
def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i,j]]+=1
    return values
hist=df(img)
plt.bar(range(256),hist)
plt.xlabel('intensity')
plt.ylabel('frequency')
plt.title('Histogram')
plt.show()
