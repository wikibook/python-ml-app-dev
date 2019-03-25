import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# 붓꽃 데이터 읽어 들이기
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
y_nums = list(map(lambda v : labels[v] , y_labels))

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_nums, train_size=0.8)

# # 붓꽃 데이터의 입력값(4차원)과 출력값(3차원)을 넣을 위치 정의하기
x  = tf.placeholder(tf.float32, [None, 4])
y_ = tf.placeholder(tf.float32, [None, 3])

# 가중치와 바이어스 변수 정의하기
w = tf.Variable(tf.zeros([4, 3])) # 가중치
b = tf.Variable(tf.zeros([3])) # 바이어스

# 소프트맥스 회귀 정의하기
y = tf.nn.softmax(tf.matmul(x, w) + b)

# 모델 훈련하기
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(cross_entropy)

# 정답률 구하기
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# 변수 초기화하기
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 변수 초기화하기
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습하기
train_feed_dict = {x: x_train, y_: y_train}
for step in range(300):
    sess.run(train, feed_dict=train_feed_dict)

# 테스트 데이터를 사용해 최종 정답률 구하기
acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
print("정답률=", acc)
