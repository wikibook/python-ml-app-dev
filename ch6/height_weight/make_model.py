import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

in_size = 2 # 체중과 키를 입력으로
nb_classes = 6 # 체형은 6단계로 구별

# MLP모델의 구조 정의하기
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

# 모델 컴파일하기
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])

model.save('hw_model.h5')
print("saved")