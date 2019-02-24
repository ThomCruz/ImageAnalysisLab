import cv2
import numpy as np

img = cv2.imread('filter.jpg')
cv2.imshow('image',img)
width, height =img.shape[:2]
print(width)
print(height)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel=[0,.125,0,0.125,0.50,0.125,0,.125,0]
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
        gray1[x,y]=0
        for i in range(9):
            sum=sum+kernel[i]*val_ar[i]
        gray1[x,y]=sum

cv2.imshow('gray1',gray1)
cv2.imwrite("output/la3_1.jpg",gray1)
