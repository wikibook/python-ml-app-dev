import cv2
import numpy as np

# 웹 카메라로부터 입력받기
cap = cv2.VideoCapture(0)
while True:
    # 이미지 추출하기
    _, frame = cap.read()
    # 이미지 축소하기
    frame = cv2.resize(frame, (500,300))
    r = frame[:, :, 2]
    img = np.zeros(r.shape, dtype=np.uint8)
    img[r > 120] = 255
    
    # 윈도우에 이미지 출력하기 --- (*2)
    cv2.imshow('RED Camera', img)
    # Enter 키가 입력되면 반복 종료하기
    if cv2.waitKey(1) == 13: break

cap.release() # 카메라 해제하기
cv2.destroyAllWindows() # 윈도우 제거하기

