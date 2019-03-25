import cv2
import numpy as np

# 카메라 입력받기
cap = cv2.VideoCapture(0)
# 동영상 출력 전용 객체 생성하기
fmt = cv2.VideoWriter_fourcc('m','p','4','v')
fps = 20.0
size = (640, 360)
writer = cv2.VideoWriter('test.m4v', fmt, fps, size) # --- (*1)

while True:
    _, frame = cap.read() # 동영상 입력
    # 이미지 축소하기
    frame = cv2.resize(frame, size)
    # 이미지 출력하기 --- (*2)
    writer.write(frame)
    # 화면에도 출력하기
    cv2.imshow('frame', frame)
    # Enter 키가 입력되면 반복 종료하기
    if cv2.waitKey(1) == 13: break
    
writer.release()
cap.release()
cv2.destroyAllWindows() # 윈도우 제거하기


