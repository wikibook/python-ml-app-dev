import cv2

# 이미지 읽어 들이기
img = cv2.imread("test.jpg")

# 이미지 저장하기
cv2.imwrite("out.png", img)

