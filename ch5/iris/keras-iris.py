import tensorflow as tf
import tensorflow.contrib.keras as keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 붓꽃 데이터 읽어 들이기 --- (*1)
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y_labels = iris_data.loc[:,"Name"]
x_data = iris_data.loc[:,["SepalLength","SepalWidth","PetalLength","PetalWidth"]]

# 레이블 데이터를 One-hot 형식으로 변환하기
labels = {
    'Iris-setosa': [1, 0, 0], 
    'Iris-versicolor': [0, 1, 0], 
    'Iris-virginica': [0, 0, 1]
}
y_nums = np.array(list(map(lambda v : labels[v] , y_labels)))
x_data = np.array(x_data)

# 학습 전용과 테스트 전용으로 분리하기 --- (*2)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_nums, train_size=0.8)

# 모델의 구조 정의하기 --- (*3)
Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax')) # ---(*3a)

# 모델 구축하기 --- (*4)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# 학습 실행하기 --- (*5)
model.fit(x_train, y_train,
    batch_size=20,
    epochs=300)

# 모델 평가하기 --- (*6)
score = model.evaluate(x_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=', score[0])
