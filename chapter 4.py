import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
#print(img)
#img[200:300,100:300]= 255,0,0 (if we need crop part of colour img)
#img[:]= 255,0,0

cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
#img.shape[0] is height, img.shape[1] is width
cv2.rectangle(img, (0, 0), (250, 350), (0, 0, 255), 2)
#cv2.FILLED IF WE WANT TO FILL THE RECTANGLE
cv2.circle(img,(400,50),30,(255,255,0),5)
cv2.putText(img,"Diptanu",(300,200),cv2.FONT_ITALIC,1.5,(0,100,255),3)


cv2.imshow("Img", img)

cv2.waitKey(0)
