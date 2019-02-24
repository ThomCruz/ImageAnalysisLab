import cv2
import numpy as np
import matplotlib.pyplot as plt
pic = cv2.imread('image2.png',0)
#pic = imageio.imread('img/parrot.jpg')
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
gray = gray(pic)

'''
log transform
-> s = c*log(1+r)

So, we calculate constant c to estimate s
-> c = (L-1)/log(1+|I_max|)

'''

max_ = np.max(gray)

def log_transform():
    return (255/np.log(1+max_)) * np.log(1+gray)

plt.figure(figsize = (5,5))
plt.imshow(log_transform(), cmap = plt.get_cmap(name = 'gray'))
plt.axis('off');
