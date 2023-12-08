# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libMapping_v12 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 300
plot=False
loadOld=False

# %%
CIRC_ID='CIRC_00438'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI_Z')
ID = os.path.dirname(dicomPath).split('\\')[-1]

mapPath=rf'{dicomPath}_p.mapping'
if os.path.exists(mapPath) and loadOld:
    MP03 = mapping(data=rf'{dicomPath}_p.mapping')
else:
    MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber',reject=False)

# %%
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.imshow_corrected(ID=f'{MP03.ID}_raw',plot=plot)
cropzone=MP03.cropzone
# %%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_T1')
#CIRC_ID='CIRC_00302'
ID = os.path.dirname(dicomPath).split('\\')[-1]

mapPath=rf'{dicomPath}_p.mapping'
if os.path.exists(mapPath) and loadOld:
    MP01 = mapping(data=rf'{dicomPath}_p.mapping')
else:
    MP01 = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327)
#%%
MP01.cropzone=cropzone
MP01.go_crop()
MP01.go_resize(scale=2)
fig,axs=MP01.imshow_corrected(ID='MP01_T1_raw',plot=plot)
#%%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2')

mapPath=rf'{dicomPath}_p.mapping'
if os.path.exists(mapPath) and loadOld:
    MP02 = mapping(data=rf'{dicomPath}_p.mapping')
else:
    MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False)
#%%
MP02.cropzone=cropzone

MP02.go_crop()
MP02.go_resize(scale=2)
MP02.imshow_corrected(ID='MP02_T2_raw',plot=plot)

#%%
print('Conducting Moco in all')
data_tmp=np.concatenate((MP01._data,MP02._data,MP03._data),axis=-1)
data_list=MP01.valueList+MP02.valueList+MP03.valueList
from copy import deepcopy
MP_All=deepcopy(MP01)
MP_All.ID='MPALL'
MP_All._raw_data=np.concatenate((MP01._raw_data,MP02._raw_data,MP03._raw_data),axis=-1)
MP_All._data=data_tmp
MP_All.valueList=data_list
data_tmp_lrt=moco(data_tmp,MP_All,data_list)
# %%
