# 라이브러리 읽어 들이기
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 학습 전용 데이터와 결과 준비하기
# X , Y
learn_data = [[0,0], [1,0], [0,1], [1,1]]
# X xor Y
learn_label = [0, 1, 1, 0]  #(*) xor 전용 레이블로 변경

# 알고리즘 지정하기(LinierSVC)
clf = LinearSVC()

# 학습전용데이터와결과학습하기 
clf.fit(learn_data, learn_label)

# 테스트 데이터로 예측하기
test_data = [[0,0], [1,0], [0,1], [1,1]]
test_label = clf.predict(test_data)

# 테스트 결과 평가하기
print(test_data , "의 예측 결과: " ,  test_label)
print("정답률 = " , accuracy_score([0, 1, 1, 0], test_label))  #(*) xor 전용 레이블로 변경