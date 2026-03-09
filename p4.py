import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. DATA
X_wine, y_wine = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X_wine, y_wine, test_size=0.3, random_state=42)
# 2. SCALE → PCA to 2D (for visualization & faster training)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
pca = PCA(n_components=2)
X_tr_p = pca.fit_transform(X_tr_s); X_te_p = pca.transform(X_te_s)

# 3. MODEL & TRAIN  (warm_start=True lets us track loss per epoch)
mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1, warm_start=True, random_state=0)
losses = []
for _ in range(100):
    mlp.fit(X_tr_p, y_tr)
    losses.append(mlp.loss_)

# 4. EVALUATE
print("\n=== Program 4: Wine Classification (Backprop) ===")
print(f"Accuracy: {mlp.score(X_te_p, y_te):.2f}")

# 5. VISUALIZE: Loss curve + Confusion Matrix
plt.plot(losses); plt.title("P4: Training Loss per Epoch")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.show()

cm = confusion_matrix(y_te, mlp.predict(X_te_p))
ConfusionMatrixDisplay(cm, display_labels=load_wine().target_names).plot(cmap='Blues')
plt.title("P4: Confusion Matrix"); plt.grid(False); plt.show()
