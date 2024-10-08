import cv2
import numpy as np

img = cv2.imread("Resources/Diptanu Das.jpg")
kernel = np.ones((5,5),np.uint8)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(9,9),0)
imgCanny = cv2.Canny(img,150,200)
imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)

cv2.imshow("Gray img",imgGray)
cv2.imshow("Blur img",imgBlur)
cv2.imshow("Canny img",imgCanny)
cv2.imshow("Dialation img",imgDialation)
cv2.imshow("Erotion img",imgEroded)
cv2.waitKey(0)
