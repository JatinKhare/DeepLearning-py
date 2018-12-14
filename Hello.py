import numpy as np
import tensorflow as tf
a = tf.constant("Sheldon ")
b = tf.constant(["Cooper"])
c = tf.add(a,b)
session = tf.Session()
result = session.run(c)
print(result)
session.close(); 

#or

with tf.Session() as session:
	result = session.run(c)
	print(result)
