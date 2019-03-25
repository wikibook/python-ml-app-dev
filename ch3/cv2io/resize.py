import matplotlib.pyplot as plt
import cv2

# 이미지 읽어 들이기
img = cv2.imread("test.jpg")
# 이미지 크기 변경하기
im2 = cv2.resize(img, (600, 300))
# 크기 변경한 이미지 저장하기
cv2.imwrite("out-resize.png", im2)

# 이미지 출력하기
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()
