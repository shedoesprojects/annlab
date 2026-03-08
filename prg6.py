import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

X = np.array([[1,1],[2,2],[3,3],[6,5],[7,8],[8,8]])
y = np.array([0,0,0,1,1,1])

model = Perceptron()
model.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y)

x_vals = np.linspace(0,10)
y_vals = -(model.coef_[0][0]*x_vals + model.intercept_)/model.coef_[0][1]

plt.plot(x_vals,y_vals)
plt.title("Linear Decision Boundary")
plt.show()