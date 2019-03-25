import matplotlib.pyplot as plt
import cv2

# 이미지 읽어 들이기
img = cv2.imread("test.jpg")
# 네거티브 반전
img = 255 - img
# 이미지 출력하기
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
