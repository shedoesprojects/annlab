import numpy as np
import matplotlib.pyplot as plt

# 1. DATA  (marks of 2 subjects → Pass=1 / Fail=0)
X = np.array([[78,82],[65,70],[40,35],[50,45],[30,25],
              [85,88],[38,42],[72,75],[29,31],[80,79]])
y = np.array([1,1,0,0,0,1,0,1,0,1])

# 2. INIT weights & bias
w = np.zeros(2)
b = 0
lr = 0.01

# 3. ACTIVATION  (step function)
def step(x): return 1 if x >= 0 else 0

# 4. TRAIN  (update rule: w += lr * error * x)
for _ in range(10000):
    for i in range(len(X)):
        pred  = step(np.dot(X[i], w) + b)
        error = y[i] - pred
        w += lr * error * X[i]
        b += lr * error

print("=== Program 1: Perceptron ===")
print("Weights:", w, "  Bias:", b)

# 5. VISUALIZE  (scatter + decision boundary)
colors = ['g' if yi==1 else 'r' for yi in y]
for i in range(len(X)):
    plt.scatter(X[i,0], X[i,1], c=colors[i])
xv = [min(X[:,0])-5, max(X[:,0])+5]
yv = [(-w[0]*x - b) / w[1] for x in xv]
plt.plot(xv, yv, label='Decision Boundary')
plt.xlabel("Sub 1"); plt.ylabel("Sub 2")
plt.title("P1: Perceptron – Pass(G) vs Fail(R)")
plt.legend(); plt.grid(True); plt.show()
