import cv2, os, copy
from sklearn.externals import joblib

# 학습한 데이터 읽어 들이기
clf = joblib.load("fish.pkl")
output_dir = "./bestshot"
img_last = None # 이전 프레임을 저장할 변수
fish_th = 3 # 이미지로 출력할 기준이 되는 물고기 수
count = 0
frame_count = 0
if not os.path.isdir(output_dir): os.mkdir(output_dir)

# 동영상 파일로부터 입력받기 --- (*1)
cap = cv2.VideoCapture("fish.mp4")
while True:
    # 이미지 추출하기
    is_ok, frame = cap.read()
    if not is_ok: break
    frame = cv2.resize(frame, (640, 360))
    frame2 = copy.copy(frame)
    frame_count += 1
    # 이전 프레임과 비교를 위해 흑백으로 변환하기 --- (*2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    if not img_last is None:
        # 차이 추출하기
        frame_diff = cv2.absdiff(img_last, img_b)
        cnts = cv2.findContours(frame_diff, 
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[1]
        # 차이가 있는 부분에 물고기가 있는지 확인하기
        fish_count = 0
        for pt in cnts:
            x, y, w, h = cv2.boundingRect(pt)
            if w < 100 or w > 500: continue # 노이즈 제거하기
            # 추출한 영역에 물고기가 있는지 확인하기 --- (*3)
            imgex = frame[y:y+h, x:x+w]
            imagex = cv2.resize(imgex, (64, 32))
            image_data = imagex.reshape(-1, )
            pred_y = clf.predict([image_data]) # --- (*4)
            if pred_y[0] == 1:
                fish_count += 1
                cv2.rectangle(frame2, (x, y), (x+w, y+h), (0,255,0), 2)
        # 물고기가 많이 있는지 확인하기 --- (*5)
        if fish_count > fish_th:
            fname = output_dir + "/fish" + str(count) + ".jpg"
            cv2.imwrite(fname, frame)
            count += 1
    cv2.imshow('FISH!', frame2)
    if cv2.waitKey(1) == 13: break
    img_last = img_b
cap.release()
cv2.destroyAllWindows()
print("ok", count, "/", frame_count)

