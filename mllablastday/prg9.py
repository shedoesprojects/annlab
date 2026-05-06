import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

data=load_iris()
xt,xte,yt,yte=train_test_split(data.data,data.target,test_size=0.3)
model=KNeighborsClassifier()
model.fit(xt,yt)
ypred=model.predict(xte)
for i in range(len(yte)):
    if ypred[i]==yte[i]:
        print(f"{i} correct : {ypred[i]}")
    else:
        print(f"{i} wrong : {ypred[i]}")


