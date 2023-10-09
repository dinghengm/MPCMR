# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v12 import readFolder,decompose_LRT
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import scipy.io as sio
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
# %%
CIRC_ID='CIRC_00356'
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI\MR000000.dcm')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP03 = mapping(data=dicomPath,ID=ID,CIRC_ID=CIRC_ID)
# %%
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.go_moco()
MP03.imshow_corrected()
cropzone=MP03.cropzone
# %%
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_T1')
#CIRC_ID='CIRC_00302'
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01 = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID)
MP01.cropzone=cropzone
#%%
MP01.go_crop()
MP01.go_resize(scale=2)
MP01.go_moco()
MP01.imshow_corrected()

#%%
#Please: Change th
MP01._delete(d=[2,3,8,-3])
MP01.go_moco()
MP01.imshow_corrected()


# %%
#Read MP02
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP02.cropzone=cropzone
# %%
MP02.go_crop()
MP02.go_resize(scale=2)
MP02.go_moco()
MP02.imshow_corrected()
# %%

#TEMP:
#Try to implement the LRT
#Use MP01._coregister_elastix()
#Perform on the first images:
#MP02-MP01
#Insert the MP01 to MP02
MP02_temp=np.concatenate((np.expand_dims(MP01._data[:,:,:,0],axis=-1),MP02._data),axis=-1)
MP02_regressed=decompose_LRT(MP02_temp)
#Show the images
#Remove the MP01
Nx,Ny,Nz,_=np.shape(MP02_temp)
MP02_temp_corrected_temp=np.copy(MP02_regressed)
for z in range(Nz):
    MP02_temp_corrected_temp[:,:,z,:]=MP02._coregister_elastix(MP02_regressed[:,:,z,:],MP02_temp[:,:,z,:])
MP02._data=MP02_temp_corrected_temp[:,:,:,1::]
MP02.imshow_corrected()
#MP03-MP01

#%%
#MP03-MP01
MP03_temp=np.concatenate((np.expand_dims(MP01._data[:,:,:,0],axis=-1),MP03._data),axis=-1)
MP03_regressed=decompose_LRT(MP03_temp)
Nx,Ny,Nz,_=np.shape(MP03_temp)
MP03_temp_corrected_temp=np.copy(MP03_regressed)
for z in range(Nz):
    MP03_temp_corrected_temp[:,:,z,:]=MP03._coregister_elastix(MP03_regressed[:,:,z,:],MP03_temp[:,:,z,:])
MP03._data=MP03_temp_corrected_temp[:,:,:,1::]
MP03.imshow_corrected()

#%%
#Get the Maps
ADC=MP03.go_cal_ADC()
MP03._map=ADC*1000
MP03.imshow_map()

#%%
MP01._save_nib()
print(f'MP01:{MP01.valueList}')
MP02._save_nib()
print(f'MP01:{MP02.valueList}')
#%%
#Load the data
path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'MP01_T1.mat'))
map_data=map_data['T1']
MP01._map= map_data

map_data=sio.loadmat(os.path.join(path,'MP02_T2.mat'))
map_data=map_data['T2']
MP02._map= map_data

# %%
#%matplotlib qt <--- you need this if you haven't turned it on in vscode
MP03.go_segment_LV(reject=None, image_type="b0")
# save
MP03.save()
# look at stats
MP03.show_calc_stats_LV()
MP03.imshow_overlay()

#%%
MP01._update_mask(MP03)
MP02._update_mask(MP03)
MP02.save()
# look at stats
MP02.show_calc_stats_LV()
MP02.imshow_overlay()
MP01.save()
# look at stats
MP01.show_calc_stats_LV()
MP01.imshow_overlay()


#%%