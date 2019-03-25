from keras.models import load_model
import numpy as np
import random
from keras.utils.np_utils import to_categorical

# 학습하기모델 읽어 들이기 --- (*1)
model = load_model('hw_model.h5')
# 학습한 데이터 읽어 들이기 --- (*2)
model.load_weights('hw_weights.h5')

# 정답 데이터를 1000개 만들기 --- (*3)
x = []
y = []
for i in range(1000):
    h = random.randint(130, 180)
    w = random.randint(30, 100)
    bmi = w / ((h / 100) ** 2)
    type_no = 1
    if bmi < 18.5:
        type_no = 0
    elif bmi < 25:
        type_no = 1
    elif bmi < 30:
        type_no = 2
    elif bmi < 35:
        type_no = 3
    elif bmi < 40:
        type_no = 4
    else:
        type_no = 5
    x.append(np.array([h / 200, w / 150]))
    y.append(type_no)

# 형식 변환하기 --- (*4)
x = np.array(x)
y = to_categorical(y, 6)
# 정답률 확인하기 --- (*5)
score = model.evaluate(x, y, verbose=1)
print("정답률=", score[1], "손실 =", score[0])

