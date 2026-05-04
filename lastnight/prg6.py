import numpy as np
import matplotlib.pyplot as plt
x=np.array([[1,2],[2,3],[3,3],[5,5],[6,6],[7,7]])
y=np.array([0,0,0,1,1,1])
w=np.zeros(2)
b=0
lr=0.1
for _ in range(20):
    for xi,yi in zip(x,y):
        yhat=1 if (np.dot(xi,w)+b)>=0 else 0
        w+=(yi-yhat)*lr*xi
        b+=(yi-yhat)*lr
print("final weights: ",w)
print("final bias: ",b)
plt.scatter(x[:,0],x[:,1],c=y)
x=np.linspace(0,8)
plt.plot(x,(-w[0]*x-b)/w[1],'k--')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()