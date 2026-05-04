import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
data=load_wine()
xt,xte,yt,yte=train_test_split(data.data,data.target,test_size=0.3)
model=MLPClassifier(hidden_layer_sizes=(10,),max_iter=1000).fit(xt,yt)
pred=model.predict(xte)
loss=model.loss_curve_
plt.plot(loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.show()
print(classification_report(yte,pred))
cm=confusion_matrix(yte,pred)
disp=ConfusionMatrixDisplay(cm, display_labels=load_wine().target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
pca=PCA(n_components=2)
x=pca.fit_transform(data.data)
plt.scatter(x[:,0],x[:,1],c=data.target)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()