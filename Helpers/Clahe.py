import cv2
import numpy as np
 
image = cv2.imread('pa.jpg')
image = cv2.resize(image, (500, 600))
 
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
clahe = cv2.createCLAHE(clipLimit = 10)
final_img = clahe.apply(image_bw)
 
# final_img = cv2.equalizeHist(image_bw)

cv2.imshow('CLAHE image', final_img)
cv2.waitKey(0)