import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

x1=np.random.normal([-1,-1],0.3,(60,2))
x2=np.random.normal([1,1],0.3,(60,2))
x=np.vstack((x1,x2))
y=np.array([0]*60+[1]*60)

xt,xte,yt,yte=train_test_split(x,y,test_size=0.3)
model=MLPClassifier(hidden_layer_sizes=(5,),max_iter=1000).fit(xt,yt)
pred=model.predict(xte)
print(classification_report(yte,pred))
plt.scatter(xte[:,0],xte[:,1],c=yte,label="test")
plt.scatter(xt[:,0],xt[:,1],c=yt,marker='x',label="train")
plt.xlabel("scaled claw size")
plt.ylabel("scaled shell hardness")
plt.legend()
plt.grid()
plt.show()

