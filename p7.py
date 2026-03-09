import numpy as np
import matplotlib.pyplot as plt
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

# 1. DATA (sine wave time series)
data = (np.sin(0.05 * np.arange(500)) + 0.1*np.random.randn(500)).reshape(-1,1)
scaler = MinMaxScaler()
data_s = scaler.fit_transform(data)

# 2. CREATE SEQUENCES  (seq_len past values → predict next)
def make_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len]); y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_seq, y_seq = make_sequences(data_s)
split = int(len(X_seq)*0.8)
X_tr, X_te = X_seq[:split].reshape(-1,10,1), X_seq[split:].reshape(-1,10,1)
y_tr, y_te = y_seq[:split], y_seq[split:]

# 3. MODEL  (LSTM layer → Dense output)
model = Sequential([LSTM(50, input_shape=(10,1)), Dense(1)])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_tr, y_tr, epochs=20, batch_size=32,
                    validation_split=0.1, verbose=0)

# 4. PREDICT & RMSE
pred = scaler.inverse_transform(model.predict(X_te))
actual = scaler.inverse_transform(y_te)
print("\n=== Program 7: LSTM ===")
print(f"RMSE: {np.sqrt(mean_squared_error(actual, pred)):.4f}")

# 5. VISUALIZE
plt.figure(figsize=(12,4))
plt.plot(data, label='Original', alpha=0.6)
plt.plot(np.arange(10+split, 10+split+len(pred)), pred, label='LSTM Prediction', color='orange')
plt.title("P7: LSTM – Time Series Prediction")
plt.xlabel("Time"); plt.ylabel("Value"); plt.legend(); plt.grid(True); plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("P7: LSTM Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.legend(); plt.grid(True); plt.show()
