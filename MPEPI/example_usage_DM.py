#%%
import numpy as np
import os

import pydicom 
from pydicom.filereader import read_dicomdir
#from CIRC_tools import imshowgrid, uigetfile, toc
from pathlib import Path
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tensorly.decomposition import tucker
import tensorly as tl
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
import time
import multiprocessing
import imageio # for gif
from roipoly import RoiPoly, MultiRoi
from matplotlib import pyplot as plt #for ROI poly and croping
from imgbasics import imcrop #for croping
from tqdm.auto import tqdm # progress bar
from ipyfilechooser import FileChooser # ui get file
import pickle # to save diffusion object
import fnmatch # this is for string comparison dicomread
import pandas as pd
from skimage.transform import resize as imresize
import nibabel as nib
import scipy.io as sio


try:
    from numba import njit #super fast C-like calculation
    _global_bNumba_support = True
except:
    print('does not have numba library ... slower calculations only')
    _global_bNumba_support = False


from MPEPI.libMapping_v12 import mapping,readFolder  

%matplotlib qt

#%%

dicomPath=r'C:\Research\MRI\MP_EPI\CIRC_00302_22737_CIRC_00302_22737\MP03_DWI_Zoom'
ID = dicomPath.split('\\')[-1]
CIRC_ID='CIRC_00302'
npy_data,valueList,dcmList= readFolder(dicomPath,sortBy='seriesNumber')
#npy_data,valueList,dcmList= readFolder(dicomPath)
#print(dcmList)
#print(valueList)
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# create instance of mapping object
#Read the object
#npy_data=r'C:\Research\MRI\MP_EPI\CIRC_00292_22737_CIRC_00292_22737\MP02_T2.mapping'




#%%
#Read Npy
map = mapping(data=npy_data,ID=ID,CIRC_ID=CIRC_ID) 
#Read Mapping
'''
map_data='MP03_DWI_nonZoom.mapping'
map = mapping(data=map_data) 
'''
#Read Nii.gz
#map=mapping(data='MP03_DWI_nonZoom_moco.nii.gz',ID=ID,CIRC_ID=CIRC_ID)

# %%
#Truncation and Resize and MOCO
map.go_crop_Auto()
print(map.shape)
map.go_resize(scale=2)
map.go_moco(method = 'lrt')
map.path=os.getcwd()




# %%
Nz=map.Nz
A1=np.copy(map._raw_data)
A2=np.copy(map._data)
for i in range(Nz):
    A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:])*255 #original data
    #A1[:,:,i,:] = map._raw_data[...,i,:]*0.6
    A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:])*255 #Processed data
    #A2[:,:,i,:] = map._data[...,i,:]*0.6
A1=map._crop_Auto(A1)
A1=map._resize(A1,newshape=np.shape(A2))
print(np.shape(A2))

for i in range(Nz):
    dataShow=np.hstack((A1[...,i,:],A2[...,i,:],A1[...,i,:]-A2[...,i,:]))
    map.createGIF(f'{map.ID}_moco_{i}.gif',dataShow,fps=5)

# %%
#Seperate them if it's the combination:
Nx,Ny,Nz,Nd=np.shape(map._data)
#save_data=np.concatenate((map._data[:,:,:,0:3],np.expand_dims(map._data[:,:,:,-1],axis=-1)),axis=-1)
#save_nii=str(f'{map.path}\{ID}_4point.nii.gz')
save_data=map._data
save_nii=str(f'{map.path}\{map.ID}_moco.nii.gz')
nib.save(nib.Nifti1Image(save_data,affine=np.eye(4)),save_nii)
save_data_raw=map._raw_data
save_nii_raw=str(f'{map.path}\{map.ID}.nii.gz')
nib.save(nib.Nifti1Image(save_data_raw,affine=np.eye(4)),save_nii_raw)
print('Saved successfully!')

#%%
#Get the T1 values
list=sorted(set(valueList))
list=[i+35 for i in list]
print(list)

# %%
#%% Sepecial save option:
save_nii=str(f'{map.path}\{ID}_moco_TI.nii.gz')
#save_data=np.concatenate((map._data[:,:,:,1:6],np.expand_dims(map._data[:,:,:,1],axis=-1)),axis=-1)
save_data=map._data[:,:,:,12::]

nib.save(nib.Nifti1Image(save_data,affine=np.eye(4)),save_nii)

#%%
#Show the images
########################################Run#######################################
root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
img_dir= os.path.join(root_dir,f'{map.CIRC_ID}_{map.ID}')
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

#%%
#For T1
map_data=sio.loadmat('MP01_T1.mat')
map_data=map_data['T1']
map._map= map_data
#Run IF T1
num_slice=map.Nz
figsize = (3.4*num_slice, 3)
#T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=[0,3000] # vmax = np.max(dti.md)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    #im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap="magma")
    im = axes[sl].imshow(np.flip(map._map[..., sl],axis=0), vmin=crange[0], vmax=crange[1], cmap="magma")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(img_dir)
plt.show()
#%%
#Reload the file:
#Get first  t2

map_data=sio.loadmat('MP02_T2.mat')
map_data=map_data['T2']


#map=mapping(data='MP02_T2_Bright.nii.gz',ID='MP02_T2_Bright.mapping',CIRC_ID='CIRC_00292')
map._map=map_data
map.path=os.getcwd()
crange=[0,150]
num_slice=map.Nz
figsize = (3.4*num_slice, 3)
#T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    #im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap="viridis")
    im = axes[sl].imshow(np.flip(map._map[..., sl],axis=0), vmin=crange[0], vmax=crange[1], cmap="viridis")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(img_dir)
plt.show()

# %%
##IF DWI: Calculation:
if 'dwi' in ID.lower():
    ADC=map.go_calc_ADC()
    np.save(str(f'{map.path}\{ID}_moco_ADC'),ADC)
#map=mapping(data='MP02_T2_Bright.nii.gz',ID='MP02_T2_Bright.mapping',CIRC_ID='CIRC_00292')
map._map=  ADC*1000

map.path=os.getcwd()
crange=[0,3]
num_slice=map.Nz
figsize = (3.4*num_slice, 3)
#T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=[0,3] # vmax = np.max(dti.md)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(np.flip(map._map[..., sl],axis=0), vmin=crange[0],vmax=crange[1], cmap="hot")
    #im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap="hot")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(img_dir)
plt.show()


#%%
map.go_segment_LV(image_type='map',crange=crange)

map.show_calc_stats_LV()
map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=crange)

#Save again
#filename=f'{map.path}\Processed\{ID}.mapping'
filename=f'{map.ID}_p.mapping'
#filename='Processed\MP03_DWI_nonZoom_moco.mapping'
with open(filename, 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(map, outp, pickle.HIGHEST_PROTOCOL)

print('Segmentation Saved')


#%%
#Read Npy
#map = mapping(data=npy_data,ID=ID,tval=valueList) 
#Read Mapping
'''
map_data='MP03_DWI_nonZoom.mapping'
map = mapping(data=map_data) 
'''
data_path=os.path.dirname(dicomPath)
T1_bssfp,_,_  = readFolder(os.path.join(data_path,'t1map_long_t1_3slice_8mm_150_gap_MOCO_T1-2'))

T1_bssfp_fb,_,_=readFolder(os.path.join(data_path,'t1map_long_t1_3slice_8mm_150_gap_free_breathing_MOCO_T1'))

T2_bssfp,_,_=readFolder(os.path.join(data_path,'t2map_flash_3slice_8mm_150_gap_MOCO_T2'))
T2_bssfp_fb,_,_=readFolder(os.path.join(data_path,'t2map_flash_3slice_8mm_150_gap_free_breathing_MOCO_T2'))




#%%
#Read clinical mapping
data=T2_bssfp_fb.squeeze()
map=mapping(data=np.expand_dims(data,axis=-1),ID='T2-FLASH-FB',CIRC_ID=CIRC_ID)

map.shape=np.shape(map._data)
map.go_crop()
map.go_resize(scale=2)
map.shape=np.shape(map._data)
cropzone=map.cropzone

#%%
#IfT1
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
map.cropzone=cropzone
print(map.shape)
map.path=os.getcwd()
map.go_segment_LV(image_type='map',crange=crange)
# %%
map.show_calc_stats_LV()
map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=crange)

#Save again
#filename=f'{map.path}\Processed\{ID}.mapping'
filename=f'{map.ID}_p.mapping'
with open(filename, 'wb') as outp:  # Overwrites any existing file.
    pickle.dump(map, outp, pickle.HIGHEST_PROTOCOL)
print('Saved the segmentation sucessfully')
# %%
plt.close()
for z in range(num_slice):
    fig,axs=plt.subplots(3,3,figsize=[15,20])

    im1=axs[0,0].imshow(T1_Bright_EPI[:,:,2],cmap='magma',vmin=0,vmax=3000)
    axs[0,0].axis('off')
    axs[0,0].set_title('MP-EPI-T1')
    im2=axs[0,1].imshow(T2_Bright_EPI_5point[:,:,2],cmap='viridis',vmin=0,vmax=150)
    axs[0,1].axis('off')
    axs[0,1].set_title('MP-EPI-T1')
    im3=axs[0,2].imshow(ADC_EPI[:,:,2]*1000,cmap='hot',vmin=0,vmax=3)
    axs[0,2].axis('off')
    axs[0,2].set_title('MP-EPI-ADC')
    axs[1,0].imshow(T1_All._data[:,:,2,0],cmap='magma',vmin=0,vmax=3000)
    axs[1,0].axis('off')
    axs[1,0].set_title('T1-MOLLI')
    axs[2,0].imshow(T1_All._data[:,:,2,1],cmap='magma',vmin=0,vmax=3000)
    axs[2,0].axis('off')
    axs[2,0].set_title('T1-MOLLI-FB')
    axs[1,1].imshow(T2_All._data[:,:,2,0]/10,cmap='viridis',vmin=0,vmax=150)
    axs[1,1].axis('off')
    axs[1,1].set_title('T2-FLASH')

    axs[2,1].imshow(T2_All._data[:,:,2,1]/10,cmap='viridis',vmin=0,vmax=150)
    axs[2,1].set_title('T2-FLASH-FB')
    axs[2,1].axis('off')
    axs[1,2].axis('off')
    axs[2,2].axis('off')

    #cb_ax=fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar1=fig.colorbar(im1,ax=axs[:,0])
    cbar2=fig.colorbar(im2,ax=axs[:,1])
    cbar3=fig.colorbar(im3,ax=axs[0,2])
    plt.show()