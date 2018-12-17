import numpy as np
import matplotlib.pyplot as plt

marks= np.array([[35],[46],[57],[92],[74],[77]])
grade = np.array([4,5,6,10,8,8])
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(marks,grade)
pred = clf.predict([[40],[98]])
real = [[6],[10]]
print(np.ceil(pred)-1)
plt.scatter(marks,grade)
plt.plot([[40],[98]],np.ceil(pred)-1,'r')
from sklearn.metrics import accuracy_score

