import cv2

a = cv2.imread('./Style2Paints/4.jpg')
# b = cv2.imread('./Sketch/4.jpg')

a = cv2.resize(a, (1024, 1400))
# b = cv2.resize(b, (1024, 1400))

cv2.imwrite('./a.jpg', a)
# cv2.imwrite('./b.png', b)