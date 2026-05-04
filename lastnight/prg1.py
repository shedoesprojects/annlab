import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,0])
y=np.array([0,1])
w=0
b=0
lr=0.1
for i in range(20):
    for xi,yi in zip(x,y):
        p=1 if w*xi+b>=0 else 1
        w+=(p-yi)*lr*xi
        b+=(p-yi)*lr
print("wts and bias: ", w , b)
n=np.linspace(-0.2,1.2)
plt.scatter(x,y)
plt.plot(n,-w*n-b,label="activation line")
plt.xlabel("input")
plt.ylabel('output')
plt.grid()
plt.show()