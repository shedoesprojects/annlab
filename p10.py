from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from tensorflow.keras.layers import GRU, Embedding
from sklearn.metrics import classification_report

# 1. DATA (binary synthetic sequences)
X_gru = np.array([[1,2,3,4,5] if i%2==0 else [5,4,3,2,1] for i in range(100)])
y_gru = np.array([1 if i%2==0 else 0 for i in range(100)])
X_g_tr, X_g_te, y_g_tr, y_g_te = train_test_split(X_gru, y_gru, test_size=0.2,
                                                    stratify=y_gru, random_state=42)

# 2. MODEL  (Embedding→GRU→Dense sigmoid)
gru_model = Sequential([
    Embedding(input_dim=10, output_dim=8),
    GRU(16),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist10 = gru_model.fit(X_g_tr, y_g_tr, epochs=20, batch_size=8,
                       validation_split=0.2, verbose=0)

# 3. EVALUATE
loss10, acc10 = gru_model.evaluate(X_g_te, y_g_te, verbose=0)
print("\n=== Program 10: GRU ===")
print(f"Test Accuracy: {acc10:.4f}")
y_pred10 = (gru_model.predict(X_g_te) > 0.5).astype(int)
print(classification_report(y_g_te, y_pred10))

# 4. VISUALIZE: Confusion Matrix + Loss/Accuracy plots
cm10 = confusion_matrix(y_g_te, y_pred10)
plt.figure(figsize=(4,3))
sns.heatmap(cm10, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
plt.title("P10: GRU Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(hist10.history['loss'], label='Train'); ax[0].plot(hist10.history['val_loss'], label='Val')
ax[0].set_title("P10: GRU Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(hist10.history['accuracy'], label='Train'); ax[1].plot(hist10.history['val_accuracy'], label='Val')
ax[1].set_title("P10: GRU Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()