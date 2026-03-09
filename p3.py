import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. DATA (synthetic crab features: ClawSize, ShellHard, Weight, Length)
np.random.seed(42)
n = 100
X_crab = np.vstack([
    np.random.multivariate_normal([2,2,1.5,1], np.diag([0.8]*4), n),  # Not Crab
    np.random.multivariate_normal([7,7,6,5],   np.diag([0.8]*4), n)   # Crab
])
y_crab = np.hstack([np.zeros(n), np.ones(n)])

# 2. SCALE + SPLIT
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_crab)
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_crab, test_size=0.3, random_state=42)

# 3. MODEL & TRAIN  (pattern net = MLP with hidden layers)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
mlp.fit(X_tr, y_tr)

# 4. EVALUATE
print("\n=== Program 3: Crab Classification ===")
print(f"Accuracy: {accuracy_score(y_te, mlp.predict(X_te)):.2f}")
print(classification_report(y_te, mlp.predict(X_te), target_names=['Not Crab','Crab']))

# 5. VISUALIZE (2D projection – first 2 features only)
plt.scatter(X_scaled[y_crab==0,0], X_scaled[y_crab==0,1], c='red',  label='Not Crab', alpha=0.6)
plt.scatter(X_scaled[y_crab==1,0], X_scaled[y_crab==1,1], c='blue', label='Crab',     alpha=0.6)
plt.xlabel("Scaled Claw Size"); plt.ylabel("Scaled Shell Hardness")
plt.title("P3: Crab Classification (2D view)")
plt.legend(); plt.grid(True); plt.show()