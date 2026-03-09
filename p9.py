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

# 1. DATA (MNIST – 28x28 handwritten digits)
(X_tr9, y_tr9), (X_te9, y_te9) = mnist.load_data()
X_tr9 = X_tr9[..., np.newaxis] / 255.0   # normalize + add channel dim
X_te9 = X_te9[..., np.newaxis] / 255.0
y_tr9_c = to_categorical(y_tr9); y_te9_c = to_categorical(y_te9)

# 2. MODEL  (Conv→Pool→Flatten→Dense→Softmax)
cnn = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist9 = cnn.fit(X_tr9, y_tr9_c, epochs=5, validation_split=0.1, verbose=0)

# 3. EVALUATE
loss9, acc9 = cnn.evaluate(X_te9, y_te9_c, verbose=0)
print("\n=== Program 9: CNN ===")
print(f"Test Accuracy: {acc9:.4f}")

# 4. VISUALIZE: Confusion Matrix + Accuracy plot
y_pred9 = np.argmax(cnn.predict(X_te9), axis=1)
cm9 = confusion_matrix(y_te9, y_pred9)
plt.figure(figsize=(7,6))
sns.heatmap(cm9, annot=True, fmt='d', cmap='Blues')
plt.title("P9: CNN Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.show()

plt.plot(hist9.history['accuracy'], label='Train'); plt.plot(hist9.history['val_accuracy'], label='Val')
plt.title("P9: CNN Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True); plt.show()