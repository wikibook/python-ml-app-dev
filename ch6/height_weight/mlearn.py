import keras
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import numpy as np
import sqlite3
import os

# 데이터베이스에서 데이터 100개 읽어 들이기 --- (*1)
dbpath = "./hw.sqlite3"
select_sql = "SELECT * FROM person ORDER BY id DESC LIMIT 100"
# 읽어 들인 데이터를 리스트에 추가하기 --- (*2)
x = []
y = []
with sqlite3.connect(dbpath) as conn:
    for row in conn.execute(select_sql):
        id, height, weight, type_no = row
        # 데이터를 정규화하기 --- (*3)
        height = height / 200
        weight = weight / 150
        y.append(type_no)
        x.append(np.array([height, weight]))

# 모델 읽어 들이기 --- (*4)
model = load_model('hw_model.h5')

# 이미 학습 데이터가 있는 경우 읽어 들이기 --- (*5)
if os.path.exists('hw_weights.h5'):
    model.load_weights('hw_weights.h5')

nb_classes = 6 # 체형은 6단계로 구별
y = to_categorical(y, nb_classes) # One-hot 벡터로 변환하기

# 학습하기 --- (*6)
model.fit(np.array(x), y,
    batch_size=50,
    epochs=100)

# 결과 저장하기 --- (*7)
model.save_weights('hw_weights.h5')

