import math
import cv2
import copy
import numpy as np 
from matplotlib import pyplot as plt 

img = cv2.imread('image2.png', 0)
bit_plane_1 = copy.copy(img)
bit_plane_2 = copy.copy(img)
bit_plane_3 = copy.copy(img)
bit_plane_4 = copy.copy(img)
bit_plane_5 = copy.copy(img)
bit_plane_6 = copy.copy(img)
bit_plane_7 = copy.copy(img)
bit_plane_8 = copy.copy(img)

for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        bit_plane_1[i,j] = img[i,j] & 1
        bit_plane_2[i,j] = img[i,j] & 2
        bit_plane_3[i,j] = img[i,j] & 4
        bit_plane_4[i,j] = img[i,j] & 8
        bit_plane_5[i,j] = img[i,j] & 16
        bit_plane_6[i,j] = img[i,j] & 32
        bit_plane_7[i,j] = img[i,j] & 64
        bit_plane_8[i,j] = img[i,j] & 128
        
fig, ax = plt.subplots(2, 4)

ax[0, 0].plot(bit_plane_1,color='gray') #row=0, col=0
ax[0, 1].plot(bit_plane_2,color='gray') #row=1, col=0
ax[0, 2].plot(bit_plane_3,color='gray') #row=0, col=1
ax[0, 3].plot(bit_plane_4,color='gray')
ax[1, 0].plot(bit_plane_5,color='gray')
ax[1, 2].plot(bit_plane_6,color='gray')
ax[1, 2].plot(bit_plane_7,color='gray')
ax[1, 3].plot(bit_plane_8,color='gray') #row=1, col=1
plt.show()
