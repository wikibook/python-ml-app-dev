# TensorFlow 읽어 들이기 --- (*1)
import tensorflow as tf

# 상수 정의하기 --- (*2)
a = tf.constant(100)
b = tf.constant(30)

# 연산 정의하기 --- (*3)
add_op = a + b

# 세션 시작하기 --- (*4)
sess = tf.Session()
res = sess.run(add_op) # 식 평가하기
print(res)

