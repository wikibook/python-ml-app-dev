import tensorflow as tf

# 상수 정의하기 --- (*1)
a = tf.constant(10)
b = tf.constant(20)
c = tf.constant(30)

# 연산 정의하기 --- (*2)
mul_op = (a + b) * c

# 세션 시작하기 --- (*3)
sess = tf.Session()
res = sess.run(mul_op)
print(res)

