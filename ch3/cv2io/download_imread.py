# 이미지 다운로드
import urllib.request as req
url = "http://uta.pw/shodou/img/28/214.png"
req.urlretrieve(url, "test.png")

# OpenCV로 읽어 들이기
import cv2
img = cv2.imread("test.png")
print(img)

