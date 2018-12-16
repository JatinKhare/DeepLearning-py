import numpy as np
import matplotlib.pyplot as plt
import tensorflow

marks= np.array([[35],[46],[83],[55],[57],[92],[74],[77]])
grade = np.array([4,5,9,6,6,10,8,8])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(marks,grade)
pred = clf.predict([[40],[98]])
real = [[6],[10]]
print(pred)
plt.scatter(marks,grade)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(real, pred)
print(accuracy*100)



