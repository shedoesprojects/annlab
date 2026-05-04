import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

t=np.arange(0,50)
data=np.sin(t/10)
x=t.reshape(-1,1,1)
model=Sequential()
model.add(GRU(10, input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x,data,epochs=200,verbose=0)
pred=model.predict(x)
plt.plot(t,pred,label="predicted")
plt.plot(t,data,label="actual")
plt.legend()
plt.show()
