import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽어 들이기 --- (*1)
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기 --- (*2)
y = iris_data.loc[:,"Name"]
x = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기 --- (*3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# 학습하기 --- (*4)
clf = SVC()
clf.fit(x_train, y_train)

# 평가하기 --- (*5)
y_pred = clf.predict(x_test)
print("정답률 = " , accuracy_score(y_test, y_pred))