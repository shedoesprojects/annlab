import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.models import Sequential

x=np.random.rand(100,28,28,1)
y=np.random.randint(0,2,100)

model=Sequential()
model.add(Conv2D(8,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(10,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="BinaryCrossentropy",metrics=['accuracy'])
history=model.fit(x,y,epochs=30,verbose=0)

plt.plot(history.history['accuracy'])
plt.xlabel("Epochs")
plt.ylabel("Acuracy")
plt.title("CNN Analysis")
plt.legend()
plt.show()