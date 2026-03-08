import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,Dense

X = np.random.rand(100,4,1)
y = np.random.rand(100)

model = Sequential()
model.add(GRU(10,input_shape=(4,1)))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

history = model.fit(X,y,epochs=20,verbose=0)

plt.plot(history.history['loss'])
plt.title("GRU Training Loss")
plt.show()