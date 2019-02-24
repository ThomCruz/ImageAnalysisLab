import cv2
import numpy as np
from numpy import pi, exp, sqrt
img = cv2.imread('house.jpg')
cv2.imshow('image',img)
width, height =img.shape[:2]
print(width)
print(height)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray2= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#gaussian with correlation

s, k = 1, 1 #  generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
kernel = np.outer(probs, probs)
kernelN=[]
for x in kernel:
    for y in x:
        kernelN.append(y)
for x in range(1,width-1):
    for y in range(1,height-1):
        val_ar = []
        val_ar.append(gray[x-1,y-1])     
        val_ar.append(gray[x-1,y])       
        val_ar.append(gray[x-1,y+1])     
        val_ar.append(gray[x,y-1])      
        val_ar.append(gray[x,y])     
        val_ar.append(gray[x,y+1])       
        val_ar.append(gray[x+1,y-1])    
        val_ar.append(gray[x+1,y])      
        val_ar.append(gray[x+1,y+1])
        sum=0
        for i in range(9):
            sum=sum+kernelN[i]*val_ar[i]
        gray[x,y]=sum
          

#laplacian with convolution, currently done with correlation, same output will come from convolution too, ass kernel is symmetric
for x in range(1,width-1):
    for y in range(1,height-1):
        filter1=gray[x-1,y-1]+gray[x-1,y+1]+gray[x-1,y]+gray[x,y-1]-8*gray[x,y]+gray[x,y+1]+gray[x+1,y-1]+gray[x+1,y]+gray[x+1,y+1]
        gray1[x,y]=0
        gray1[x,y]=filter1

        filter2=-1*gray[x-1,y-1]+2*gray[x,y-1]-1*gray[x+1,y-1]+2*gray[x-1,y]-4*gray[x,y]+2*gray[x,y+1]-1*gray[x+1,y-1]+2*gray[x+1,y]-1*gray[x+1,y+1]
        gray2[x,y]=0
        gray2[x,y]=filter2
           

cv2.imshow('gray1',gray1)
cv2.imwrite("output/la3_2_out1.jpg",gray1)
cv2.imshow('gray2',gray2)
cv2.imwrite("output/la3_2_out2.jpg",gray2)
cv2.waitKey(0)
cv2.destroyAllWindows()  
