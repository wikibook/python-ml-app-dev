import cv2
import numpy as np

# 웹 카메라로부터 입력받기 --- (*1)
cap = cv2.VideoCapture(0)
while True:
    # 카메라의 이미지 읽어 들이기 --- (*2)
    _, frame = cap.read()
    # 이미지를 축소해서 출력하기 --- (*3)
    frame = cv2.resize(frame, (500,300))
    # 윈도우에 이미지 출력하기 --- (*4)
    cv2.imshow('OpenCV Web Camera', frame)
    # ESC 또는 Enter 키가 입력되면 반복 종료하기
    k = cv2.waitKey(1) # 1msec 대기
    if k == 27 or k == 13: break

cap.release() # 카메라 해제하기
cv2.destroyAllWindows() # 윈도우 제거하기

