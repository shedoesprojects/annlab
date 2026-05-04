import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

x1=np.random.normal([-1,-1],0.3,(60,2))
x2=np.random.normal([1,1],0.3,(60,2))
x=np.vstack((x1,x2))
y=np.array([0]*60+[1]*60)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
model=MLPClassifier(hidden_layer_sizes=(5,),max_iter=1000).fit(xtrain,ytrain)
pred=model.predict(xtest)
print(classification_report(ytest,pred))
plt.scatter(xtest[:,0],xtest[:,1],c=ytest,label="Test")
plt.scatter(xtrain[:,0],xtrain[:,1],c=ytrain,marker='x',label="training data")
plt.title("crab classification")
plt.xlabel("scaled claw size")
plt.ylabel("scaled shell hardness")
plt.legend()
plt.grid()
plt.show()