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
from tqdm.auto import tqdm # progress bar

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 300
# %%
CIRC_ID='CIRC_00373'
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI\MR000000.dcm')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP03 = mapping(data=dicomPath,ID=ID,CIRC_ID=CIRC_ID)
#MP03 = mapping(data=r'C:\Research\MRI\MP_EPI\CIRC_00356_22737_CIRC_00356_22737\MP03_DWI_p.mapping')
# %%
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.go_moco()
MP03.imshow_corrected(ID=f'{MP03.ID}_raw')
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
fig,axs=MP01.imshow_corrected(ID='MP01_T1_Truncated_1')
for i in range(np.shape(axs)[-1]):
    axs[0,i].set_title(f'{i}',fontsize=5)
img_dir= os.path.join(os.path.dirname(MP01.path),f'{MP01.CIRC_ID}_{MP01.ID}_Original')
plt.savefig(img_dir)
#%%
#Please: subtract the one you don't want
#MP01._delete(d=[0,2,3,4,5,7,10,11,-4,-7])
MP01._delete(d=[2,3,5,6,7,8,12,13,20,21,23,28,29,31,32,35,36,37])
MP01.go_moco()
fig,axs=MP01.imshow_corrected(ID='MP01_T1_Truncated_2')

# %%
#Read MP02
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP02.cropzone=cropzone
# %%
MP02.go_crop()
MP02.go_resize(scale=2)
MP02.go_moco()
MP02.imshow_corrected(ID='MP02_T2_raw')
# %%

#TEMP:
#Try to implement the LRT
#Use MP01._coregister_elastix()
#Perform on the first images:
#MP02-MP01
#Insert the MP01 to MP02
print('Conducting Moco in MP02')
MP02_temp=np.concatenate((np.expand_dims(MP01._data[:,:,:,0],axis=-1),MP02._data),axis=-1)
MP02_regressed=decompose_LRT(MP02_temp)
#Show the images
#Remove the MP01
Nx,Ny,Nz,_=np.shape(MP02_temp)
MP02_temp_corrected_temp=np.copy(MP02_regressed)
for z in range(Nz):
    MP02_temp_corrected_temp[:,:,z,:]=MP02._coregister_elastix(MP02_regressed[:,:,z,:],MP02_temp[:,:,z,:])
MP02._data=MP02_temp_corrected_temp[:,:,:,1::]
MP02.imshow_corrected(ID='MP02_T2_Combined')

#%%
#MP03-MP01
print('Conducting Moco in MP03')
MP03_temp=np.concatenate((np.expand_dims(MP01._data[:,:,:,0],axis=-1),MP03._data),axis=-1)
MP03_regressed=decompose_LRT(MP03_temp)
Nx,Ny,Nz,_=np.shape(MP03_temp)
MP03_temp_corrected_temp=np.copy(MP03_regressed)
for z in range(Nz):
    MP03_temp_corrected_temp[:,:,z,:]=MP03._coregister_elastix(MP03_regressed[:,:,z,:],MP03_temp[:,:,z,:])
MP03._data=MP03_temp_corrected_temp[:,:,:,1::]
MP03.imshow_corrected(ID='MP03_Combined')
generateVolumeGIFwithRaw(MP01)
generateVolumeGIFwithRaw(MP02)
generateVolumeGIFwithRaw(MP03)
#%%
#Get the Maps
ADC=MP03.go_cal_ADC()
MP03._map=ADC*1000
MP03.imshow_map()

#%%
MP01._save_nib()
print(f'MP01:{MP01.valueList}')
MP02._save_nib()
print(f'MP02:{MP02.valueList}')
#%%
#Load the data

path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'MP01_T1.mat'))
map_data=map_data['T1']
MP01._map= map_data
MP01.imshow_map()
map_data=sio.loadmat(os.path.join(path,'MP02_T2.mat'))
map_data=map_data['T2']
MP02._map= map_data
MP02.imshow_map()
#%%
#PLOT MOCO
Nz=MP01.Nz
A2=np.copy(MP01._data)
for i in range(Nz):
    A2[:,:,i,:] = MP01._data[...,i,:]/np.max(MP01._data[...,i,:])*255

for i in range(Nz):
    MP01.createGIF(f'{MP01.ID}_moco_{i}.gif',A2[:,:,i,:],fps=5)
# %%
#%matplotlib qt <--- you need this if you haven't turned it on in vscode
MP03.go_segment_LV(reject=None, image_type="b0")
# save
MP03.save()
# look at stats
MP03.show_calc_stats_LV()
MP03.imshow_overlay()

#%%

MP02._update_mask(MP03)
MP02.save()
# look at stats
MP02.show_calc_stats_LV()
MP02.imshow_overlay()
MP01._update_mask(MP03)
MP01.save()
# look at stats
MP01.show_calc_stats_LV()
MP01.imshow_overlay()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3])


#%%
#Show the T1 T2 Map
data_path=os.path.dirname(dicomPath)
T1_bssfp,_,_  = readFolder(os.path.join(data_path,r'MR t1map_long_t1_3slice_8mm_150_gap_MOCO_T1-2'))

T1_bssfp_fb,_,_=readFolder(os.path.join(data_path,r'MR t1map_long_t1_3slice_8mm_150_gap_free_breathing_MOCO_T1-3'))

T2_bssfp,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_3slice_8mm_150_gap_MOCO_T2'))
T2_bssfp_fb,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_3slice_8mm_150_gap_free_breathing_MOCO_T2-2'))

#%%
data=T2_bssfp_fb.squeeze()
map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH_FB',CIRC_ID=CIRC_ID)

map.shape=np.shape(map._data)
map.go_crop()
map.go_resize(scale=2)
map.shape=np.shape(map._data)
cropzone=map.cropzone
#%%
#################################Second Run:
#data=T2_bssfp.squeeze()
data=T1_bssfp_fb.squeeze()
map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI_FB',CIRC_ID=CIRC_ID)
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)
map.cropzone=cropzone
map.shape=np.shape(map._data)
map.go_crop()
map.go_resize(scale=2)
map.shape=np.shape(map._data)
#%%
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
#If T2
#map._map=data_crop
from skimage.transform import resize as imresize
data_crop=imresize(data_crop,np.shape(map._data))
crange=[0,3000]
map.crange=crange
map._map=data_crop.squeeze()
map.cropzone=cropzone
print(map.shape)
map.path=os.getcwd()
map.go_segment_LV(image_type='map',crange=crange)
# %%
#Truncation and Resize and MOCO
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map._data))
#If T2
#map._map=data_crop
map._map=data_crop.squeeze()/10
crange=[0,150]
map.crange=crange
map.cropzone=cropzone
print(map.shape)
map.path=os.getcwd()
map.go_segment_LV(image_type='map',crange=crange)
# %%
map.show_calc_stats_LV()
map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=crange)

#Save again
#filename=f'{map.path}\Processed\{ID}.mapping'
map.save()
print('Saved the segmentation sucessfully')
# %%
