import tensorflow as tf
sess = tf.Session()
msg  = tf.constant('Hello')
print(sess.run(msg).decode('utf-8'))

