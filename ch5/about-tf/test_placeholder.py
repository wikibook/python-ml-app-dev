import tensorflow as tf

# 플레이스홀더 정의하기 --- (*1)
a = tf.placeholder(tf.int32, [5])

# 벡터를 2배 하는 연산 정의하기 --- (*2)
two = tf.constant(2)
x2_op = a * two

# 세션 시작하기 --- (*3)
sess = tf.Session()

# 플레이스 홀더에 값을 넣어 실행하기 --- (*4)
res1 = sess.run(x2_op, feed_dict={ a: [1, 2, 3, 4, 5] })
print(res1)
res2 = sess.run(x2_op, feed_dict={ a: [5, 6, 7, 10, 100] })
print(res2)


