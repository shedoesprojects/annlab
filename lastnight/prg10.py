import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

t=np.arange(0,50)
y=np.sin(t/10)
x=t.reshape(-1,1,1)
model=Sequential()
model.add(GRU(10,input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x,y,epochs=200,verbose=0)
pred=model.predict(x)
plt.plot(t,y,label="actual")
plt.plot(t,pred,label="predicted")
plt.title("GRU analysis")
plt.legend()
plt.show()
