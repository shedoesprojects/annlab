import matplotlib.pyplot as plt
import numpy as np

x=np.array([[0,1],[1,0],[0,0],[1,1]])
yand=np.array([0,0,0,1])
yor=np.array([1,1,0,1])

def fin(y):
    w=np.zeros(2)
    b=0
    lr=0.1
    for i in range(20):
        for xi,yi in zip(x,y):
            p=1 if np.dot(w,xi)+b>=0 else 0
            w+=lr*(yi-p)*xi
            b+=lr*(yi-p)
    return w,b

w1,b1=fin(yand)
w2,b2=fin(yor)
print(f"AND: weight: {w1} bias: {b1}")
print(f"OR: weight: {w2} bias: {b2}")
n=np.linspace(-0.2,1.2)
plt.scatter(x[:,0],x[:,1],c=yand,label="AND")
plt.plot(n,(-w1[0]*n-b1)/w1[1],'k--',label="decision boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()

plt.scatter(x[:,0],x[:,1],c=yor,label="OR")
plt.plot(n,(-w2[0]*n-b2)/w2[1],'k--',label="decision boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid()
plt.show()