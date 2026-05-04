import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

t=np.arange(0,50)
data=np.sin(t)
x=[]
y=[]
for i in range(len(data)-3):
    x.append(data[i:i+3])
    y.append(data[i+3])
x=np.array(x)
y=np.array(y)
x=x.reshape(x.shape[0],x.shape[1],1)
model=Sequential()
model.add(LSTM(10,activation="relu",input_shape=(1,1)))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x,y,epochs=200,verbose=0)
pred=model.predict(x)
plt.plot(pred,label="predicted")
plt.plot(y,label="actual")
plt.title("LSTM Analysis")
plt.legend()
plt.show()