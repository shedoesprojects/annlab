import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2],[2,3],[3,3],[5,5],[6,6],[7,7])
y=np.array([0,0,1,1,1,1])
w=np.zeros(2)
b=0
lr=0.1
for i,j in zip(z,y):
    pred=1 if np.dot(w,i)+b>0 else 0
    w+=lr*(j-pred)*i
    b+=lr*(j-pred)
print("weights:",w)
print("bias:",b)

plt.scatter(x[:,0],x[:,1],c=y)
plt.title("Perceptron Decision Boundary")
plt.xlabel("Feature 1") 
plt.ylabel("Feature 2")
plt.show()