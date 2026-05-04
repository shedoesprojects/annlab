import numpy as np
import matplotlib.pyplot as plt
x=np.array([0,1])
y=np.array([1,0])
w,b,lr=0,0,0.1
for m in range(10):
    for i,j in zip(x,y):
        pred=1 if w*i+b>=0 else 0
        w+=lr*(j-pred)*i
        b+=lr*(j-pred)
print("weight",w)
print("bias",b)
plt.scatter(x,y,color='r',label='target output')
n=np.linspace(-0.2,1.2)
plt.plot(n,w*n+b,label='activation line"')
plt.xlabel('input')
plt.ylabel('output')
plt.legend()
plt.grid()
plt.show()