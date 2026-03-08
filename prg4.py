import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = load_wine()
X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(10,10),max_iter=500)
model.fit(X_train,y_train)

print("Accuracy:",model.score(X_test,y_test))

plt.plot(model.loss_curve_)
plt.title("Backpropagation Training Loss")
plt.show()