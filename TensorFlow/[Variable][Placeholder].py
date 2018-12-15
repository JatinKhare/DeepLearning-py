import tensorflow as tf

# Variable 
a = tf.Variable(31)
b = tf.constant(100)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init_op)
	print(session.run(a))

#Place Holder
a = tf.placeholder(tf.float32)

b = a/2;

with tf.Session() as session:
	result = session.run(b,feed_dict= {a:22}) 
	print(result)
