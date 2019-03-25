from keras.models import load_model
import numpy as np

# 학습하기모델 읽어 들이기 --- (*1)
model = load_model('hw_model.h5')
# 학습한 데이터 읽어 들이기 --- (*2)
model.load_weights('hw_weights.h5')
# 레이블
LABELS = [
    '저체중', '표준 체중 ', '1비만(1도)',
    '비만(2도)', '비만(3도)', '비만(4도)' 
]

# 테스트 데이터 지정하기 --- (*3)
height = 160
weight = 50
# 정규화하기 --- (*4)
test_x = [height / 200, weight / 150]
# 예측하기 --- (*5)
pre = model.predict(np.array([test_x]))
idx = pre[0].argmax()
print(LABELS[idx], '/ 가능성', pre[0][idx])

