from detect_zip import *
import matplotlib.pyplot as plt

from sklearn.externals import joblib

# 학습한 데이터 읽어 들이기
clf = joblib.load("digits.pkl")

# 이미지에서 영역 읽어 들이기
cnts, img = detect_zipno("hagaki1.png")

# 읽어 들인 데이터 출력하기
for i, pt in enumerate(cnts):
    x, y, w, h = pt
    # 윤곽으로 감싼 부분을 작게 만들기
    x += 8
    y += 8
    w -= 16
    h -= 16
    # 이미지 데이터 추출하기
    im2 = img[y:y+h, x:x+w]
    # 데이터를 학습에 적합하게 변환하기
    im2gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) # 그레이스케일
    im2gray = cv2.resize(im2gray, (8, 8)) # 크기 변경
    im2gray = 15 - im2gray // 16 # 흑백 반전
    im2gray = im2gray.reshape((-1, 64)) # 차원 변환
    # 데이터 예측하기
    res = clf.predict(im2gray)
    # 출력하기
    plt.subplot(1, 7, i + 1)
    plt.imshow(im2)
    plt.axis("off")
    plt.title(res)

plt.show()
