import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,100)
y=np.sin(x)+np.random.rand(100)*0.2
f=np.convolve(y,[1,0,-1],mode='same')

plt.plot(y,label="Input Signal")
plt.plot(f,label="Feature Map")
plt.title("CNN Feature Extraction Analysis")
plt.legend()
plt.show()