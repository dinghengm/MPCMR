#########################
#########This is the script to read the clinical data from subject with LGE
#########FROM saved_ims_v2_Dec_14_2023
#########TO  saved_ims_v2_Dec_14_2023,CIRC_ID
#########TO STATS: mapping_Dec
#########SUBJECTS 438 and 488
#########MAPSAVEDAS overlay images

# %% ####################################################################################
%matplotlib inline
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import readFolder,decompose_LRT,moco,moco_naive
import numpy as np
import matplotlib.pyplot as plt
import os
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
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Jan_12_2024')
#%%
#CIRC_ID='CIRC_00435'
CIRC_ID='CIRC_00438'
img_save_dir=os.path.join(img_root_dir,CIRC_ID)
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
#%%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
data_path=os.path.dirname(dicomPath)
from imgbasics import imcrop
from skimage.transform import resize as imresize
#%%
#T1_bssfp,_,_  = readFolder(os.path.join(data_path,r'MR t1map_long_t1_saxs_MOCO_T1'))
T1_bssfp,_,_  = readFolder(os.path.join(data_path,r'MR t1map_long_t1_MOCO_T1'))

#T1_bssfp_post,_,_=readFolder(os.path.join(data_path,r'MR t1map_short_t1_HHR_MOCO_T1-4'))
T1_bssfp_post,_,_=readFolder(os.path.join(data_path,r'MR t1map_short_t1_HHR_MOCO_T1-4_Post'))

#T2_bssfp,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_saxs_MOCO_T2'))
T2_bssfp,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_MOCO_T2-2'))


lge_post,_,_=readFolder(os.path.join(data_path,r'MR de_trufi_overview_psir SAX moco FS_MOCO_AVG_MAG'))
#lge_post,_,_=readFolder(os.path.join(data_path,r'MR de_trufi_overview_psir SAX moco FS_b2_MOCO_AVG_MAG'))

#%%  Read T1
%matplotlib qt

data=T1_bssfp.squeeze()
map_T1=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI_Long',CIRC_ID=CIRC_ID)

map_T1._update()
map_T1.go_crop()
#map_T1.go_resize(scale=2)
map_T1._update()
cropzone=map_T1.cropzone
#%%
%matplotlib qt
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map_T1.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map_T1._data))
crange=[0,3000]
map_T1.crange=crange
map_T1._map=data_crop.squeeze()
map_T1.cropzone=cropzone
print(map_T1.shape)
map_T1.path=dicomPath
map_T1.cmap="magma"
map_T1.go_segment_LV(image_type='map',crange=crange,cmap="magma",roi_names=['endo', 'epi','septal','lateral'])
map_T1.show_calc_stats_LV()
#%%  Read T1 post
data=T1_bssfp_post.squeeze()
map_T1_post=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI_post',CIRC_ID=CIRC_ID)
map_T1_post.cropzone=cropzone
map_T1_post.shape=np.shape(map_T1_post._data)
map_T1_post.go_crop()
#map_T1_post.go_resize(scale=2)
map_T1_post.shape=np.shape(map_T1_post._data)
#%%
%matplotlib qt
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map_T1_post.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map_T1_post._data))
crange=[0,1000]
map_T1_post.crange=crange
map_T1_post._map=data_crop.squeeze()
map_T1_post.cropzone=cropzone
print(map_T1_post.shape)
map_T1_post.path=dicomPath
map_T1_post.cmap="magma"
map_T1_post.go_segment_LV(image_type='map',crange=crange,cmap="magma")
map_T1_post.show_calc_stats_LV()
#%% Read T2
data=T2_bssfp.squeeze()
map_T2=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)
map_T2.cropzone=cropzone
map_T2.shape=np.shape(map_T2._data)
map_T2.go_crop()
#map_T2.go_resize(scale=2)
#map_T1.go_resize(scale=2)
map_T2._update()
cropzone=map_T2.cropzone


#%%
%matplotlib qt
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map_T2.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map_T2._data))
#If T2
#map._map=data_crop
map_T2._map=data_crop.squeeze()/10
crange=[0,150]
map_T2.crange=crange
map_T2.cropzone=cropzone
print(map_T2.shape)
map_T2.path=dicomPath
map_T2.cmap="viridis"
map_T2.go_segment_LV(image_type='map',crange=crange,cmap="viridis")
map_T2.show_calc_stats_LV()


#%%





# %%
%matplotlib inline
for map in [map_T1,map_T1_post,map_T2]:
    # create images images per map type
    num_slice = map.Nz
    figsize = (3.4*num_slice, 3)
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=map.crange
    cmap=map.cmap

    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
    plt.savefig(os.path.join(img_save_dir, f"{map.ID}"))
    plt.show()

    map.show_calc_stats_LV()
    map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Dec.csv',crange=crange)
    map.save(filename=os.path.join(img_root_dir,f'{map.CIRC_ID}_{map.ID}_p.mapping'))
    print('Saved the segmentation sucessfully')
    #Save again
    map.save()
    print('Saved the segmentation sucessfully')
    map.imshow_overlay(path=img_save_dir,ID=f'{map.ID}_overlay',plot=plot)

# %%
# %%

%matplotlib qt

data=lge_post.squeeze()
map_lge=mapping(data=np.expand_dims(data,axis=-1),ID='LGE_SA',CIRC_ID=CIRC_ID)
map_lge._update()
map_lge.go_crop()
#map_lge.go_resize(scale=2)
map_lge._update()
#%%
slices= range(map_lge.Nz)

num_average=len(slices)
map_lge._data=map_lge._data.squeeze()
figsize = (3.4*num_average, 3)
vmin=np.min(map_lge._data)
vmax=np.max(map_lge._data)
fig, axes = plt.subplots(nrows=1, ncols=num_average, figsize=figsize, constrained_layout=True)
for ind,sl in enumerate(slices):
    axes[ind].set_axis_off()
    im = axes[ind].imshow(map_lge._data[..., sl],  cmap='gray',vmin=-400,vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)

if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_lge.ID}_all"))
plt.show()
#%%
#slices= range(map_lge.Nz)
slices=[4,8,11]
num_average=len(slices)
map_lge._data=map_lge._data.squeeze()
figsize = (3.4*num_average, 3)
vmin=np.min(map_lge._data)
vmax=np.max(map_lge._data)
fig, axes = plt.subplots(nrows=1, ncols=num_average, figsize=figsize, constrained_layout=True)
for ind,sl in enumerate(slices):
    axes[ind].set_axis_off()
    im = axes[ind].imshow(map_lge._data[..., sl],  cmap='gray',vmin=-400,vmax=vmax)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)

if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_lge.ID}_4"))
    #plt.savefig(os.path.join(img_save_dir, f"{map_lge.ID}_all"))
plt.show()

# %%
#Read clinical data
img_save_dir=os.path.join(img_root_dir,CIRC_ID)
mapList=[]
for dirpath,dirs,files in  os.walk(img_save_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('MOLLI_p.mapping') or path.endswith('FLASH_p.mapping') or path.endswith('MOLLI_post_p.mapping'):
                mapList.append(path)
print(mapList)
# %%
map_T1=mapping(mapList[0])
map_T1_post=mapping(mapList[1])
map_T2=mapping(mapList[2])
#%%
%matplotlib qt
for map in [map_T1,map_T2]:
    crange=map.crange
    cmap=map.cmap
    map.go_segment_LV(image_type='map',crange=crange,cmap=cmap,roi_names=['septal','lateral'])
#%%
map_T1_post.go_segment_LV(image_type='map',crange=map_T1_post.crange,cmap=map_T1_post.cmap,roi_names=['endo','epi','septal','lateral'])

#%%
%matplotlib inline
for map in [map_T1,map_T1_post,map_T2]:
    # create images images per map type
    num_slice = map.Nz
    figsize = (3.4*num_slice, 3)
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=map.crange
    cmap=map.cmap

    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
    plt.savefig(os.path.join(img_save_dir, f"{map.ID}"))
    plt.show()

    map.show_calc_stats_LV()
    map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Dec.csv',crange=crange)
    map.save(filename=os.path.join(img_root_dir,f'{map.CIRC_ID}_{map.ID}_plus200_p.mapping'))
    print('Saved the segmentation sucessfully')
    #Save again
    map.save()
    print('Saved the segmentation sucessfully')
    map.imshow_overlay(path=img_save_dir,ID=f'{map.ID}_overlay',plot=plot)

# %%
