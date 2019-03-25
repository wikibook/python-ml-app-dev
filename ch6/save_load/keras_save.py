from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

# 붓꽃 데이터 읽어 들이기
iris = datasets.load_iris()
in_size = 4
nb_classes=3
# 레이블 데이터를 One-hot 형식으로 변환하기
x = iris.data
y = to_categorical(iris.target, nb_classes)

# 모델 정의하기 --- (*1)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
# 컴파일하기 --- (*2)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
# 학습 실행하기 --- (*3)
model.fit(x, y, batch_size=20, epochs=50)

# 모델 저장하기 --- (*4)
model.save('iris_model.h5')
# 학습한 가중치 데이터 저장하기 --- (*5)
model.save_weights('iris_weight.h5')

