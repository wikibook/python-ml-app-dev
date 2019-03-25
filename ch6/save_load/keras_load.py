from sklearn import datasets
import keras
from keras.models import load_model
from keras.utils.np_utils import to_categorical

# 붓꽃 데이터 읽어 들이기
iris = datasets.load_iris()
in_size = 4
nb_classes=3
# 레이블 데이터를 One-hot 형식으로 변환하기
x = iris.data
y = to_categorical(iris.target, nb_classes)

# 모델 읽어 들이기 --- (*1)
model = load_model('iris_model.h5')
# 가중치 데이터 읽어 들이기 --- (*2)
model.load_weights('iris_weight.h5')

# 모델 평가하기 --- (*3)
score = model.evaluate(x, y, verbose=1)
print("정답률=", score[1])

