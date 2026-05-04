import numpy as np
import matplotlib.pyplot as plt

x=np.array([[0,0],[0,1],[1,0],[1,1]])
yand=np.array([0,0,0,1])
yor=np.array([0,1,1,1])
def fin(y):
    w=np.zeros(2)
    b=0
    lr=0.1
    for _ in range(20):
        for i,j in zip(x,y):
            pred=1 if np.dot(w,i)+b>=0 else 0
            w+=lr*(j-pred)*i
            b+=lr*(j-pred)
    return w,b
w1,b1=fin(yand)
w2,b2=fin(yor)
print(f"and gate: weight: {w1} and bias: {b1}")
print(f"or gate: weight: {w2} and bias: {b2}")

m=np.linspace(-0.2,1.2)
plt.scatter(x[:,0],x[:,1],c=yand)
plt.plot(m,(-w1[0]*m-b1)/w1[1],'k--')
plt.title("AND Gate Decision Boundary")
plt.xlabel('input 1')
plt.ylabel('input 2')   
plt.grid()
plt.show()
plt.scatter(x[:,0],x[:,1],c=yor)
plt.plot(m,(-w2[0]*m-b2)/w2[1],'k--')
plt.title("OR Gate Decision Boundary")
plt.xlabel('input 1')
plt.ylabel('input 2')   
plt.grid()
plt.show()