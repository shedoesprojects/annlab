import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

t = np.arange(0,50)
y = np.sin(t/10)

X = t.reshape(-1,1,1)   # samples, timesteps, features

model = Sequential()
model.add(GRU(10, input_shape=(1,1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X,y,epochs=200,verbose=0)

pred = model.predict(X)

plt.plot(t,y,label="Actual")
plt.plot(t,pred,label="Predicted")
plt.title("Real GRU Analysis")
plt.legend()
plt.show()