import cv2
from sklearn.externals import joblib

def predict_digit(filename):
  # 학습한 데이터 읽어 들이기
    clf = joblib.load("digits.pkl")
    # 직접 그린 손글씨 이미지 읽어 들이기
    my_img = cv2.imread(filename)
    # 이미지 데이터를 학습에 적합하게 변환하기
    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    my_img = cv2.resize(my_img, (8, 8))
    my_img = 15 - my_img // 16 # 흑백 반전
    # 2차원 배열을 1차원 배열로 변환하기
    my_img = my_img.reshape((-1, 64))
    # 데이터 예측하기
    res = clf.predict(my_img)
    return res[0]

# 이미지 파일을 지정해서 실행하기
n = predict_digit("my2.png")
print("my2.png = " + str(n))
n = predict_digit("my4.png")
print("my4.png = " + str(n))


