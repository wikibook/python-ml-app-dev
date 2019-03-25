import matplotlib.pyplot as plt
import cv2

# 이미지 읽어 들이기
img = cv2.imread("test.jpg")
# 색공간을 그레이스케일로 변환하기
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지 출력하기
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
