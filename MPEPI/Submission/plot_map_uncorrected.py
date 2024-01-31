#%%
%matplotlib inline
import os
import sys
sys.path.append('../') 
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import readFolder,decompose_LRT,moco,moco_naive
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
import pandas as pd
from skimage.transform import resize as imresize
import h5py
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
#%%
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Dec_14_2023')


#%%
CIRC_ID_List=['446','452','429','419','407','405','398','382','381','373']
CIRC_NUMBER=CIRC_ID_List[0]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
CIRC_ID='CIRC_00446'
img_root_dir = os.path.join(defaultPath, "DataSet",CIRC_ID)
saved_img_root_dir=os.path.join(defaultPath, "DataSet",CIRC_ID,"save_imgs")
img_save_dir=saved_img_root_dir
if not os.path.exists(saved_img_root_dir):
            os.mkdir(saved_img_root_dir)

mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            if CIRC_ID in os.path.basename(path):
                mapList.append(path)
T1_FB=mapping(mapList[-4])
T1=mapping(mapList[-3])
T2_FB=mapping(mapList[-2])
T2=mapping(mapList[-1])
print(mapList[-4::])

#%%
T2._data=T2._raw_data/10
T2.go_crop()
T2._map=T2._data.squeeze()
T2.imshow_map(path=saved_img_root_dir)
#%%
T1._data=T1._raw_data
T1.go_crop()
T1._map=T1._data.squeeze()
T1.imshow_map(path=saved_img_root_dir)
#%%
MP01=mapping(mapList[0])
MP02=mapping(mapList[1])
MP03=mapping(mapList[2])
# %%
%matplotlib qt
MP01._data=MP01._map[:,:,:,np.newaxis]
MP01.go_crop()
# %%
Cropzone=MP01.cropzone
MP02._data=MP02._map[:,:,:,np.newaxis]
MP02.cropzone=Cropzone
MP02.go_crop()
MP03._data=MP03._map[:,:,:,np.newaxis]
MP03.cropzone=Cropzone
MP03.go_crop()

# %%
%matplotlib inline
MP01._map=MP01._data.squeeze()
MP01.imshow_map(path=saved_img_root_dir)
MP02._map=MP02._data.squeeze()
MP02.imshow_map(path=saved_img_root_dir)
MP03._map=MP03._data.squeeze()
MP03.imshow_map(path=saved_img_root_dir)
# %%
%matplotlib inline
for map in [T1,T2,MP01,MP02,MP03]:
    # create images images per map type
    num_slice = map.Nz
    figsize = (3.4*num_slice, 3)
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=map.crange
    cmap=map.cmap

    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
    #plt.savefig(os.path.join(img_save_dir, f"{map.ID}"))
    plt.show()
# %%
for map in [MP01,MP02,MP03]:
    # create images images per map type
    num_slice = map.Nz
    figsize = (3.4*num_slice, 3)
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=map.crange
    cmap=map.cmap

    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)

    #plt.savefig(os.path.join(img_save_dir, f"{map.ID}"))
    plt.show()
#%%
#
SliceInd=2
temp=np.shape(MP02._data)
shape=(temp[0], temp[1], MP02.shape[2])
raw_volume=np.zeros((temp[0], temp[1]))
raw_volume=imcrop(MP02.mask_lv,Cropzone)
num_slice = 3
MP01_2=mapping(mapList[0])
MP02_2=mapping(mapList[1])
MP03_2=mapping(mapList[2])
for map in [MP01_2,MP02_2,MP03_2]:
    map.cropzone=Cropzone
    map.go_crop()
base_sum=np.concatenate([MP02_2._data[:, :, SliceInd, 0:6],MP03_2._data[:, :, SliceInd,  0:6]],axis=-1)
base_im = np.mean(base_sum,axis=-1)
#base_sum=np.array([MP02_2._data[:, :, SliceInd, 0:6],MP03_2._data[:, :, SliceInd,  0:6]],axis=-1)
#base_im = np.mean(base_sum,axis=0)
#%%
figsize = (3.4*num_slice, 3)
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
for ind,map in enumerate([MP01,MP02,MP03]):
    axes[ind].set_axis_off()
    axes[ind].imshow(base_im, cmap="gray", vmax=np.max(base_im*0.8))
    axes[ind].imshow(map._map[..., SliceInd], alpha=raw_volume[..., SliceInd]*1.0,vmin=map.crange[0], vmax=map.crange[1], cmap=map.cmap)
plt.show()

#%%
#Show the map with MPLLI in single slice
SliceInd=2
figsize = (3.4*2, 3*2)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, constrained_layout=True)
ax=axes.ravel()
for ind,map in enumerate([MP01,T1,MP02,T2]):
    ax[ind].set_axis_off()
    ax[ind].imshow(map._map[..., SliceInd], vmax=map.crange[1], cmap=map.cmap)
    
    #cbar=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)

plt.show()
# %%
