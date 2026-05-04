import matplotlib.pyplot as plt
import numpy as np
x=np.array([[1,2],[2,3],[3,3],[5,5],[6,6],[7,7]])
y=np.array([0,0,0,1,1,1])
w=np.zeros(2)
b=0
lr=0.1
for i in range(20):
    for xi,yi in zip(x,y):
        p=1 if (np.dot(w,xi)+b)>=0 else 0
        w+=lr*(yi-p)*xi
        b+=lr*(yi-p)
print("final weight and bias ", w, b)
n=np.linspace(0,8)
plt.scatter(x[:,0],x[:,1],c=y)
plt.plot(n,(-w[0]*n-b)/w[1],'k--')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()