import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = cv2.imread('lenna.png')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()
# Shape of image in terms of pixels. 
(rows, cols) = img.shape[:2] 

# getRotationMatrix2D creates a matrix needed for transformation. 
# We want matrix for rotation w.r.t center to 45 degree without scaling. 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
N = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
res1 = cv2.warpAffine(img, M, (cols, rows)) 
res2 = cv2.warpAffine(img, N, (cols, rows)) 

# Write image back to disk. 
cv2.imwrite('output/la5_result1.jpg', res1)
cv2.imwrite('output/la5_result2.jpg', res2) 
