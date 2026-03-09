import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. DATA (XOR – NOT linearly separable)
X_xor = np.array([[0,0],[0,1],[1,0],[1,1],[0.2,0.8],[0.8,0.2]])
y_xor = np.array([0,1,1,0,1,1])

# 2. MODEL  (1 hidden layer of 2 neurons solves XOR)
clf = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                    solver='lbfgs', max_iter=10000, random_state=0)
clf.fit(X_xor, y_xor)

print("\n=== Program 6: Linearly Separable / XOR ===")
print("Predictions:", clf.predict(X_xor))
print("Accuracy:   ", clf.score(X_xor, y_xor))

# 3. VISUALIZE decision boundary
xx, yy = np.meshgrid(np.linspace(-0.2,1.2,300), np.linspace(-0.2,1.2,300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=1, alpha=0.5, cmap='bwr')
plt.scatter(X_xor[:,0], X_xor[:,1], c=y_xor, s=100, edgecolors='k', cmap='bwr')
plt.title("P6: XOR with MLP (2 Hidden Neurons)")
plt.xlabel("X1"); plt.ylabel("X2"); plt.grid(True); plt.show()
