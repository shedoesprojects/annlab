import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Sample dataset
X = np.random.rand(200,4)
y = np.random.randint(0,2,200)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(5,),max_iter=500)
model.fit(X_train,y_train)

acc = model.score(X_test,y_test)
print("Accuracy:",acc)

plt.plot(model.loss_curve_)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()