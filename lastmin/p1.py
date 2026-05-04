import numpy as np
import matplotlib.pyplot as plt

x=np.array([0,1])
y=np.array([1,0])
w=0
b=0
lr=0.1
for i in range(20):
    for xi,yi in zip(x,y):
        p=1 if np.dot(w,xi)+b>=0 else 1
        w+=(yi-p)*lr*xi
        b+=(yi-p)*lr
print(f"Final weight {w} and bias {b}")

n=np.linspace(-0.2,1.2)
plt.scatter(x,y)
plt.plot(n,(w*n+b),'k--',label="activation line")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron")
plt.legend()
plt.grid()
plt.show()