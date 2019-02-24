import cv2
import numpy as np
cap= cv2.VideoCapture('C:\\Users\\Chandramani\\Downloads\\265725189-198338737-c39e3830-21a6-40d6-b137-9ef572146acd.mp4')
i=0
while True:
    ret, frame = cap.read()
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()
