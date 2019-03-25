import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import numpy as np

# TF-IDF 데이터베이스 읽어 들이기 --- (*1)
data = pickle.load(open("text/genre.pickle", "rb"))
y = data[0] # 레이블
x = data[1] # TF-IDF

# 학습 전용과 테스트 전용으로 구분하기 --- (*2)
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2)

# 나이브 베이즈로 학습하기 --- (*3)
model = GaussianNB()
model.fit(x_train, y_train)

# 평가하고 결과 출력하기 --- (*4)
y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
rep = metrics.classification_report(y_test, y_pred)

print("정답률=", acc)
print(rep)


