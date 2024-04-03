# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline                     
from libMapping_v13 import *  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
#%%
boolenTest=input('Would you like to save you Plot? "Y" and "y" to save')
if boolenTest.lower() == 'y':
    plot=True
else:
    plot=False


# %%
#Please try to change to CIRC_ID
CIRC_ID='CIRC_00488'
img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'CIRC_00488_1')
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'test')
if not os.path.exists(saved_img_root_dir):
    os.mkdir(saved_img_root_dir)
#Read the MP01-MP03
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            
            #if not path.endswith('m.mapping') and not path.endswith('p.mapping'):
                mapList.append(path)
print(mapList)

MP01=mapping(mapList[0])
MP02=mapping(mapList[1])
MP03=mapping(mapList[2])




# %%
Nx,Ny,Nz,_=MP02.shape
rgb_image=np.zeros((Nx,Ny,3))
for ss in range(MP02.Nz):
    r=MP01._map[...,ss]/3000 #r
    g=MP02._map[...,ss]/150 #g
    b=MP03._map[...,ss]/3 #b
    
    for x in [r,g,b]:
        x[x > 1] = 1
    print(r,b,g)
    rgb_image=(np.dstack((r,g,b)) * 255.999) .astype(np.uint8)
    plt.figure()
    plt.imshow(rgb_image,vmin=0,vmax=256)
    plt.axis('off')
    plt.show()
    plt.close()
# %%
CIRC_ID='CIRC_00498'
img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'WITH8000',CIRC_ID)
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'test')
if not os.path.exists(saved_img_root_dir):
    os.mkdir(saved_img_root_dir)
#Read the MP01-MP03
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            
            #if not path.endswith('m.mapping') and not path.endswith('p.mapping'):
                mapList.append(path)
print(mapList)

MP01=mapping(mapList[0])
MP02=mapping(mapList[1])
MP03=mapping(mapList[2])




# %%
Nx,Ny,Nz,_=MP02.shape
rgb_image=np.zeros((Nx,Ny,3))
for ss in range(MP02.Nz):
    r=MP01._map[...,ss]/3000 #r
    g=MP02._map[...,ss]/150 #g
    b=MP03._map[...,ss]/3 #b
    
    for x in [r,g,b]:
        x[x > 1] = 1
    print(r,b,g)
    rgb_image=(np.dstack((r,g,b)) * 255.999) .astype(np.uint8)
    plt.figure()
    plt.imshow(rgb_image,vmin=0,vmax=256)
    plt.axis('off')
    plt.show()
    plt.close()

# %%
