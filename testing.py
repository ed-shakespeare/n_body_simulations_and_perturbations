import numpy as np

x = (1,2,3,4)
y = (5,6,7,8)
x_,y_ = np.meshgrid(x,y)
print(x_)
print(y_)