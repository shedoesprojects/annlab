import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

X = np.random.rand(100, 28, 28, 1)
y = np.random.randint(0, 2, 100)
# Build CNN Model
model = Sequential()
model.add(Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# Train model
history = model.fit(X, y, epochs=10, verbose=0)
# Plot training accuracy
plt.plot(history.history['accuracy'])
plt.title("CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()