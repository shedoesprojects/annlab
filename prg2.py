import numpy as np

def perceptron(X, y):
    w = np.zeros(2)
    b = 0
    lr = 0.1

    def step(z):
        return 1 if z>=0 else 0

    for epoch in range(10):
        for i in range(len(X)):
            z = np.dot(X[i],w)+b
            y_pred = step(z)
            error = y[i]-y_pred
            w += lr*error*X[i]
            b += lr*error
    return w,b

X = np.array([[0,0],[0,1],[1,0],[1,1]])

# AND
y_and = np.array([0,0,0,1])
w,b = perceptron(X,y_and)

print("AND Gate Output")
for x in X:
    print(x, "->", 1 if np.dot(x,w)+b>=0 else 0)

# OR
y_or = np.array([0,1,1,1])
w,b = perceptron(X,y_or)

print("\nOR Gate Output")
for x in X:
    print(x, "->", 1 if np.dot(x,w)+b>=0 else 0)