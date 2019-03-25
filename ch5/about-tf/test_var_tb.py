import tensorflow as tf

# 변수 정의하기 --- (*1)
v = tf.Variable(0, name='v')

# 상수 정의하기
a = tf.constant(10, name='10')
b = tf.constant(20, name='20')

# 연산 정의하기 --- (*2)
mul_op = tf.multiply(a, b, name='mul')
assign_op = tf.assign(v, mul_op)

# 세션 시작하기 --- (*3)
sess = tf.Session()
# 연산 실행하기
sess.run(assign_op)

# 텐서보드로 그래프 출력하기 --- (*4)
tf.summary.FileWriter('./logs', sess.graph)

# 결과 추출하기 --- (*5)
res = sess.run(v)
print(res)


