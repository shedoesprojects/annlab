from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. DATA & MODEL
X_jh = np.array([[1.2,1.1],[1.5,1.3],[1.8,0.4],[2.0,1.7]])
y_jh = np.sin(X_jh[:,0]) + np.cos(X_jh[:,1])
model = MLPRegressor(hidden_layer_sizes=(5,), activation='tanh',
                     solver='lbfgs', max_iter=5000).fit(X_jh, y_jh)

# 2. JACOBIAN  (1st derivative – central difference)
def jacobian(f, x, h=1e-5):
    return np.array([(f(x + h*e) - f(x - h*e)) / (2*h)
                     for e in np.eye(len(x))])

# 3. HESSIAN  (2nd derivative – central difference)
def hessian(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i, ei in enumerate(np.eye(n)):
        for j, ej in enumerate(np.eye(n)):
            H[i,j] = (f(x+h*(ei+ej)) - f(x+h*(ei-ej))
                     -f(x-h*(ei-ej)) + f(x-h*(ei+ej))) / (4*h**2)
    return H

# 4. COMPUTE at x0
f  = lambda x: model.predict(x.reshape(1,-1))[0]
x0 = np.array([0.5, 0.5])
J  = jacobian(f, x0)
H  = hessian(f, x0)

print("\n=== Program 5: Jacobian & Hessian ===")
print("Jacobian:", J)
print("Hessian:\n", H)

# 5. VISUALIZE as heatmaps
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].bar(['∂f/∂x1','∂f/∂x2'], J, color=['steelblue','coral'])
axes[0].set_title("P5: Jacobian at x0=[0.5,0.5]"); axes[0].grid(True)
im = axes[1].imshow(H, cmap='coolwarm', aspect='auto')
axes[1].set_title("P5: Hessian Matrix")
axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['x1','x2']); axes[1].set_yticklabels(['x1','x2'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f'{H[i,j]:.3f}', ha='center', va='center', color='black')
plt.colorbar(im, ax=axes[1]); plt.tight_layout(); plt.show()

