import numpy as np
import cv2, pickle
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop 
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 데이터 파일과 이미지 크기 지정하기
data_file = "./png-etl1/katakana.pickle"
im_size = 25
out_size = 46 # 일본어 가타카나 문자 수(출력 수)
im_color = 1 # 이미지의 색공간 / 그레이스케일
in_shape = (im_size, im_size, im_color)

# 저장한 이미지 데이터 읽어 들이기 --- (*1)
data = pickle.load(open(data_file, "rb"))
# 이미지 데이터를 0-1 사이의 값으로 정규화하기 --- (*2)
y = []
x = []
for d in data:
    (num, img) = d
    img = img.astype('float').reshape(
      im_size, im_size, im_color) / 255
    y.append(keras.utils.np_utils.to_categorical(num, out_size))
    x.append(img)
x = np.array(x)
y = np.array(y)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# CNN모델의 구조 정의하기 --- (*3)
model = Sequential()
model.add(Conv2D(32,
          kernel_size=(3, 3),
          activation='relu',
          input_shape=in_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(out_size, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])

# 학습하고 평가하기--- (*4)
hist = model.fit(x_train, y_train,
          batch_size=128, 
          epochs=12,
          verbose=1,
          validation_data=(x_test, y_test))
# 모델 평가하기
score = model.evaluate(x_test, y_test, verbose=1)
print('정답률=', score[1], 'loss=', score[0])

# 학습 상태를 그래프로 그리기 --- (*5)
# 정답률 추이를 그래프로 그리기
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 손실 추이를 그래프로 그리기
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

