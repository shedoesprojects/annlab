import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

x=np.linspace(-2,2,100)
y=x**3+2*x**2
j=3*x**2+4*x
h=6*x+4
# Confusion Matrix
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
plt.plot(y,label="func")
plt.plot(j,label="Jacobian")
plt.plot(h,label="Hessian") 
plt.legend()
plt.title("Function, Jacobian, and Hessian Analysis")
plt.show()