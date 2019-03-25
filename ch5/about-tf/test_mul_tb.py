import tensorflow as tf

# 상수 정의하기 --- (*1)
a = tf.constant(10, name='10')
b = tf.constant(20, name='20')
c = tf.constant(30, name='30')

# 연산 정의하기 --- (*2)
add_op = tf.add(a, b, name='add')
mul_op = tf.multiply(add_op, c, name='mul')

# 세션 시작하기 --- (*3)
sess = tf.Session()
res = sess.run(mul_op)
print(res)

# 텐서보드로 그래프 출력하기 --- (*4)
tf.summary.FileWriter('./logs', sess.graph)

