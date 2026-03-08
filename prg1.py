import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])   # AND logic

# Initialize weights
w = np.zeros(2)
b = 0
lr = 0.1

# Activation
def step(z):
    return 1 if z >= 0 else 0

# Training
for epoch in range(10):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = step(z)
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

print("Weights:", w)
print("Bias:", b)

# Visualization
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Perceptron Classification")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


