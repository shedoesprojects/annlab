from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

data=load_iris()
xt,xte,yt,yte=train_test_split(data.data,data.target,test_size=0.3)
model=KNeighborsClassifier(n_neighbors=3)
model.fit(xt,yt)
pred=model.predict(xte)
for i in range(len(yte)):
    if pred[i]==yte[i]:
        print("Correc : ", pred[i])
    else:
        print("Wrong, predicted : ",pred[i],", actual : ",yte[i])
print("Model accuracy : ", accuracy_score(yte,pred))
