#This script is to create the 3d plot for MRS acquisition
#further it might be suitable for a 3d display for your code



#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_x=[35, 45, 55, 60, 72, 80, 90]  #TE
_y=[0, 70, 600, 1200, 1800]   #TI
z=[0, 50, 100,200, 400,600, 800, 1000] # bvalue


#%%

#Plot
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top=np.zeros(np.shape(x))
for ind,xi in enumerate(x):
    if xi<=60:
        top[ind]=5
    elif xi<=72:
        top[ind]=400
    elif xi>=72:
        top[ind]=1000


bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')



plt.show()
# %%
