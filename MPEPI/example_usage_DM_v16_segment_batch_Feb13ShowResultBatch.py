###############Batch contour for paper
##############From  saved_ims_v2_Feb_5_2024/NULL
##############TO  saved_ims_v2_Feb_5_2024/NULL/xxxp.mapping
# ##############Please draw on DWI
############## Can change later for other.

#%%
import argparse
import sys
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
import pandas as pd
import h5py
import warnings #we know deprecation may show bc we are using a stable older ITK version
defaultPath= r'C:\Research\MRI\MP_EPI'
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
#%%
CIRC_ID_List=[446,452,429,419,405,398,382,381,373,457,471,472,486,498,500]
#CIRC_NUMBER=CIRC_ID_List[9]
CIRC_NUMBER=CIRC_ID_List[-1]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','NULL',f'{CIRC_ID}')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('.segmentation'):
            mapList.append(path)
print(mapList)
Slice0=segmentation(mapList[0])
Slice1=segmentation(mapList[1])
Slice2=segmentation(mapList[2])


#%%
%matplotlib qt
MP01_List=[MP01_0,MP01_1,MP01_2]
slice_List=[]
for ss,map01 in enumerate(MP01_List):
    data0MP02=MP02._data[:,:,ss,:]
    data0MP03=MP03._data[:,:,ss,:]
    data0MP02_raw=MP02._raw_data[:,:,ss,:]
    data0MP03_raw=MP03._raw_data[:,:,ss,:]

    dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
    dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
    #dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
    #dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
    bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
    
    Slice=segmentation(data=dataSlice0,rawdata=dataSlice0_raw,bval=MP03.bval,bvec=MP03.bvec,CIRC_ID=CIRC_ID,ID=f'Slice{ss}',valueList=bval0)

    Slice.go_segment_LV(brightness=0.6)
    #14
    Slice.save(path=img_save_dir,ID=f'{Slice.CIRC_ID}_{Slice.ID}')
    slice_List.append(Slice)
#%%





# %%
#MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))
MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}.mapping'))
MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}.mapping'))

# %%
