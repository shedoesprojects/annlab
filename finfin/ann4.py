import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

data=load_wine()
xtrain,xtest,ytrain,ytest=train_test_split(data.data,data.target,test_size=0.3)
model=MLPClassifier(hidden_layer_sizes=(10,),max_iter=100).fit(xtrain,ytrain)
pred=model.predict(xtest)
loss=model.loss_curve_
plt.plot(loss)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")  
plt.show()
print(classification_report(ytest,pred))
print("accuracy is:",model.score(xtest,ytest))
cm=confusion_matrix(ytest,pred)
disp=ConfusionMatrixDisplay(cm,display_labels=load_wine().target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
pca=PCA(n_components=2)
x2=pca.fit_transform(data.data)
plt.scatter(x2[:,0],x2[:,1],c=data.target)
plt.title("Decision Boundary for wine classification")
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()