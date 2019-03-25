import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 데이터 읽어 들이기 --- (*1)
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")
# 학습 전용과 테스트 전용으로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1)

# y 레이블 변경하기 --- (*2)
newlist = []
for v in list(y):
    if v <= 4:
        newlist += [0]
    elif v <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

# 학습 전용과 테스트 전용으로 분리하기 --- (*3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 학습하기 --- (*4)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 평가하기 --- (*5)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("정답률=", accuracy_score(y_test, y_pred))