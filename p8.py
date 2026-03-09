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
from tensorflow.keras.layers import SimpleRNN

# 1. DATA (same sine wave setup as LSTM)
data2 = (np.sin(0.05*np.arange(500)) + 0.1*np.random.randn(500)).reshape(-1,1)
data2_s = MinMaxScaler().fit_transform(data2)
scaler2 = MinMaxScaler().fit(data2)
data2_s = scaler2.transform(data2)
X2, y2 = make_sequences(data2_s)
split2 = int(len(X2)*0.8)
X2_tr = X2[:split2].reshape(-1,10,1); X2_te = X2[split2:].reshape(-1,10,1)
y2_tr = y2[:split2]; y2_te = y2[split2:]

# 2. MODEL  (SimpleRNN layer → Dense)
rnn = Sequential([SimpleRNN(50, activation='relu', input_shape=(10,1)), Dense(1)])
rnn.compile(optimizer='adam', loss='mse')
hist2 = rnn.fit(X2_tr, y2_tr, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

# 3. EVALUATE
pred2 = scaler2.inverse_transform(rnn.predict(X2_te))
act2  = scaler2.inverse_transform(y2_te)
print("\n=== Program 8: RNN ===")
print(f"RMSE: {np.sqrt(mean_squared_error(act2, pred2)):.4f}")

# 4. VISUALIZE
plt.figure(figsize=(12,4))
plt.plot(data2, label='Original', alpha=0.6)
plt.plot(np.arange(10+split2, 10+split2+len(pred2)), pred2, label='RNN Prediction', color='green')
plt.title("P8: RNN – Time Series Prediction")
plt.xlabel("Time"); plt.ylabel("Value"); plt.legend(); plt.grid(True); plt.show()

plt.plot(hist2.history['loss'], label='Train'); plt.plot(hist2.history['val_loss'], label='Val')
plt.title("P8: RNN Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.legend(); plt.grid(True); plt.show()