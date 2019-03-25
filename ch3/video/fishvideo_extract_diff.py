import cv2, os

img_last = None # 이전 프레임을 저장할 변수
no = 0 # 이미지 장 수
save_dir = "./exfish" # 저장 디렉터리 이름
os.mkdir(save_dir) # 디렉터리 만들기

# 동영상 파일로부터 입력받기 --- (*1)
cap = cv2.VideoCapture("fish.mp4")
while True:
    # 이미지 추출하기
    is_ok, frame = cap.read()
    if not is_ok: break
    frame = cv2.resize(frame, (640, 360))
    # 흑백 이미지로 변환하기 --- (*2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    # 차이 확인하기
    if not img_last is None:
        frame_diff = cv2.absdiff(img_last, img_b) # --- (*3)
        cnts = cv2.findContours(frame_diff, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[1]
        # 차이가 있는 부분을 파일로 출력하기 --- (*4)
        for pt in cnts:
            x, y, w, h = cv2.boundingRect(pt)
            if w < 100 or w > 500: continue # 노이즈 제거하기
            # 추출한 영역 저장하기
            imgex = frame[y:y+h, x:x+w]
            outfile = save_dir + "/" + str(no) + ".jpg"
            cv2.imwrite(outfile, imgex)
            no += 1
    img_last = img_b
cap.release()
print("ok")
