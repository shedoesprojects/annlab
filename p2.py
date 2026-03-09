import numpy as np
import matplotlib.pyplot as plt

X_gate = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and  = np.array([0,0,0,1])
y_or   = np.array([0,1,1,1])

# 2. TRAIN FUNCTION (reusable perceptron)
def train_perceptron(X, y, lr=0.1, epochs=100):
    w = np.zeros(X.shape[1]); b = 0
    for _ in range(epochs):
        for xi, t in zip(X, y):
            pred  = int(np.dot(xi, w) + b >= 0)
            error = t - pred
            w += lr * error * xi
            b += lr * error
    return w, b

w_and, b_and = train_perceptron(X_gate, y_and)
w_or,  b_or  = train_perceptron(X_gate, y_or)

print("\n=== Program 2: AND / OR Gates ===")
print("AND Weights:", w_and, "Bias:", b_and)
print("OR  Weights:", w_or,  "Bias:", b_or)

# 3. PLOT both boundaries side by side
fig, axes = plt.subplots(1, 2, figsize=(10,4))
for ax, w, b, title, y in [
    (axes[0], w_and, b_and, "AND Gate", y_and),
    (axes[1], w_or,  b_or,  "OR Gate",  y_or)]:
    ax.scatter(X_gate[:,0], X_gate[:,1], c=y, cmap='bwr', s=100, edgecolors='k')
    xv = np.linspace(-0.5, 1.5, 100)
    ax.plot(xv, (-b - w[0]*xv) / w[1], 'k--', label='Boundary')
    ax.set_title(title); ax.set_xlim(-0.5,1.5); ax.set_ylim(-0.5,1.5)
    ax.grid(True); ax.legend()
plt.suptitle("P2: AND & OR Gate Decision Boundaries")
plt.tight_layout(); plt.show()
