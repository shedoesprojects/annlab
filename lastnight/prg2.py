import matplotlib.pyplot as plt
import numpy as np

x=np.array([[0,0],[0,1],[1,0],[1,1]])
yand=np.array([0,0,0,1])
yor=np.array([0,1,1,1])
def fin(y):
    w=np.zeros(2)
    b=0
    lr=0.1
    for i in range(20):
        for xi,yi in zip(x,y):
            pred=1 if np.dot(w,xi)+b>=0 else 0
            w+=(yi-pred)*lr*xi
            b+=(yi-pred)*lr
    return w,b
w1,b1=fin(yand)
w2,b2=fin(yor)
print("final w and b for and: ",fin(yand))
print("final w and b for or: ", fin(yor))

n=np.linspace(-0.1,1)
plt.scatter(x[:,0],x[:,1],c=yand)
plt.plot(n,(-w1[0]*n-b1)/w1[1],'k--')
plt.xlabel("x1")
plt.ylabel("x1")
plt.title("and")
plt.show()
plt.scatter(x[:,0],x[:,1],c=yor)
plt.plot(n,(-w2[0]*n-b2)/w2[1],'k--')
plt.xlabel("x1")
plt.ylabel("x1")
plt.title("or")
plt.show()