#%%
#This file is to fix the issue in 405
from libMapping_v14 import mapping
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
defaultPath= r'C:\Research\MRI\MP_EPI'
CIRC_ID=f'CIRC_00{int(405)}'
print(f'Running{CIRC_ID}')

img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","WITH8000",CIRC_ID)
img_save_dir=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000",CIRC_ID)
if os.path.exists(img_save_dir) is False:
    os.makedirs(img_save_dir)

#Read the MP01-MP03
mapList=[]
#I think I also need the raw data:
#Or I can use the mask from MP02
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)

MP01_0=mapping(mapList[1])
MP01_1=mapping(mapList[2])
MP01_2=mapping(mapList[3])
MP02=mapping(mapList[4])
# %%

MP03=mapping(r'C:\Research\MRI\MP_EPI\CIRC_00405_22737_CIRC_00405_22737\CIRC_RESEARCH CIRC Research\MR ep2d_MP03_DWI_DELAY50')
MP03.bval=np.array([50,50,50,500,500,500])
saved_img_root_dir= os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","WITH8000",CIRC_ID)
#All slice is wrong, need to swap
def swapSlice(obj):
    Nx,Ny,Nz,Nd= np.shape(obj._data)
    data=obj._data
    datatmp=np.zeros((Nx,Ny,Nd))
    
    #datatmp=np.copy(data)
    #First slice is the last slice
    datatmp=np.copy(data[:,:,2,:])
    data[:,:,2,:]=data[:,:,0,:]
    data[:,:,0,:]=datatmp
    obj._data=data
    return obj

MP03_copy=swapSlice(MP03)

MP03_copy.go_crop_Auto()
#MP03_copy.go_resize()
#This of course would not match with previous data.
MP03_copy.go_moco('naive')
MP03_copy.go_resize()
MP03_copy.imshow_corrected(ID=f'MP03_with8000',valueList=MP03_copy.bval,plot=plot,path=saved_img_root_dir)

# %%
%matplotlib qt

MP03_copy.go_cal_ADC()
MP03_copy.imshow_map()
#%%

MP03_copy.go_segment_LV(image_type='b0')
# %%
