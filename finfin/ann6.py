#6. 6. Write a program to implement classification of Linearly Separable Data with a perceptron.
import numpy as np
import matplotlib.pyplot as plt

X=np.array([[1,2],[2,3],[3,3],[5,5],[6,6],[7,7]])
y=np.array([0,0,0,1,1,1])
w=np.zeros(2)
b=0 
lr=0.1

for _ in range(20):
    for xi,yi in zip(X,y):
        yhat=1 if np.dot(xi,w)+b>=0 else 0
        w+=lr*(yi-yhat)*xi; b+=lr*(yi-yhat)

print("Final Weights:",w); print("Final Bias:",b)

plt.scatter(X[:,0],X[:,1],c=y)
x=np.linspace(0,8)
plt.plot(x,(-w[0]*x-b)/w[1],'k--')
plt.title("Classification of Linearly Separable Data using Perceptron")
plt.xlabel("X1"); plt.ylabel("X2")
plt.show()