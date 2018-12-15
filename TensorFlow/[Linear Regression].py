import tensorflow as tf
import numpy as np 
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Creating Data
a = 12 
b = 4
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*a + b
fig1, ax1 = plt.subplots()
plt.plot(x_data, y_data)

# adding noise 
y_data = np.vectorize(lambda y: y+ np.random.normal(loc = 0.0, scale = 0.1))(y_data)
fig2, ax1 = plt.subplots()
plt.plot(x_data, y_data,'r')


