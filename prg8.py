import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN,Dense

X = np.random.rand(100,5,1)
y = np.random.rand(100)

model = Sequential()
model.add(SimpleRNN(10,input_shape=(5,1)))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')

history = model.fit(X,y,epochs=20,verbose=0)

plt.plot(history.history['loss'])
plt.title("RNN Training Loss")
plt.show()