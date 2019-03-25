import numpy as np
import cv2, pickle
from sklearn.model_selection import train_test_split
import keras

# 데이터 파일과 이미지 크기 지정하기 --- (*1)
data_file = "./png-etl1/katakana.pickle"
im_size = 25
in_size = im_size * im_size
out_size = 46 # 일본어 가타카나 문자 수

# 저장한 이미지 데이터 읽어 들이기 --- (*2)
data = pickle.load(open(data_file, "rb"))

# 이미지 데이터를 0-1 사이의 값으로 정규화하기 --- (*3)
y = []
x = []
for d in data:
    (num, img) = d
    img = img.reshape(-1).astype('float') / 255
    y.append(keras.utils.np_utils.to_categorical(num, out_size))
    x.append(img)
x = np.array(x)
y = np.array(y)

# 학습 전용과 테스트 전용 분리하기 --- (*4)
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# 모델의 구조 정의하기 --- (*5)
Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dense(out_size, activation='softmax'))

# 모델 컴파일하고 학습 실행하기 --- (*6)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
model.fit(x_train, y_train,
    batch_size=20, epochs=50, verbose=1,
    validation_data=(x_test, y_test))

# 모델 평가하기 --- (*7)
score = model.evaluate(x_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=', score[0])

