import cv2

img = cv2.imread("Resources/Diptanu Das.jpg")
print(img.shape)

imgResize = cv2.resize(img,(290,444))
print(imgResize.shape)

imgCropped = img[0:350,100:290]

cv2.imshow("Img",img)
#cv2.imshow("Img resize",imgResize)
cv2.imshow("Img CROPPED",imgCropped)

cv2.waitKey(0)
