import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

X = np.random.rand(100,3,1)
y = np.random.rand(100)

model = Sequential()
model.add(LSTM(10,input_shape=(3,1)))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

history = model.fit(X,y,epochs=20,verbose=0)

plt.plot(history.history['loss'])
plt.title("LSTM Training Loss")
plt.show()