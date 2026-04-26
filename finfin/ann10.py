import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import confusion_matrix

t = np.arange(0,50)
y = np.sin(t/10)

X = t.reshape(-1,1,1)   # samples, timesteps, features

model = Sequential()
model.add(GRU(10, input_shape=(1,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X,y,epochs=200,verbose=0)

pred = model.predict(X)

# Convert to binary classes
y_true = (y > 0).astype(int)           # cf actual: positive sine = 1, negative = 0
y_pred = (pred.flatten() > 0).astype(int)  # cf predicted: same logic

print("Confusion Matrix") # cf
print(confusion_matrix(y_true, y_pred)) #cf

plt.plot(t,y,label="Actual")
plt.plot(t,pred,label="Predicted")
plt.title("Real GRU Analysis")
plt.legend()
plt.show()