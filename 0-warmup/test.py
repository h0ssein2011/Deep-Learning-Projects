def Relu(z):
    return max(0 ,z)


# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt


x=list(range(-20 , 20))
y =[Relu(item) for item in x]

print(x[:5])
print(y[:5])
plt.plot(x,y)
plt.show()