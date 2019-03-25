# 다운로드한 이미지 출력하기
import matplotlib.pyplot as plt
import cv2
img = cv2.imread("test.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

