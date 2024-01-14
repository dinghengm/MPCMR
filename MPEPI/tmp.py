#%%
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
from CIRC_tools import *
matplotlib.rcParams['savefig.dpi'] = 400
plot=False
#%%
CIRC_ID_List=['446','452','429','419','407','405','398','382','381','373']
#CIRC_NUMBER=CIRC_ID_List[9]
CIRC_NUMBER=CIRC_ID_List[0]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Jan_12_2024')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            if CIRC_ID in os.path.basename(path):
                mapList.append(path)
map_T1=mapping(mapList[0])
map_T2=mapping(mapList[1])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
map_DWI=mapping(mapList[2])
#%%
def imshowMap(obj,path,plot):
    num_slice=obj.Nz
    volume=obj._map
    ID=str('map_' + obj.CIRC_ID + '_' + obj.ID)
    crange=obj.crange
    cmap=obj.cmap
    figsize = (3.4*num_slice, 3)

    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    axes=axes.ravel()
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(volume[..., sl],vmin=crange[0],vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.018, aspect=8)
    img_dir= os.path.join(path,f'{ID}')
    if plot:
        plt.savefig(img_dir)
    pass
#%%
%matplotlib inline
map_T1_post.crange=[0,1600]
img_save_dir=os.path.join(img_root_dir,CIRC_ID)
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
imshowMap(obj=map_T1,plot=plot,path=img_save_dir)
imshowMap(obj=map_T2,plot=plot,path=img_save_dir)
imshowMap(obj=map_DWI,plot=plot,path=img_save_dir)
imshowMap(obj=map_T1_post,plot=plot,path=img_save_dir)

#%%
%matplotlib qt
map_T1.go_crop()
map_T1.go_resize(scale=2)
cropzone=map_T1.cropzone
#%%
map_T2.cropzone=cropzone
map_T2.go_crop()
map_T2.go_resize(scale=2)
map_DWI.cropzone=cropzone
map_DWI.go_crop()
map_DWI.go_resize(scale=2)
#%%
#Crop the map and the data
for map in [map_T1,map_T2,map_DWI]:
    Nz=map.Nz
    data=map._map
    from imgbasics import imcrop
    temp = imcrop(data[:,:,0], cropzone)
    shape = (temp.shape[0], temp.shape[1], Nz)
    data_crop = np.zeros(shape)
    for z in tqdm(range(map.Nz)):
        data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
    data_crop=imresize(data_crop,np.shape(map._data)[0:3])
    map._map=data_crop.squeeze()
    map._update()
    print(map.shape)

#%%
%matplotlib qt
map_DWI.go_segment_LV(reject=None,z=[0,1,2], image_type="b0_avg",roi_names=['endo', 'epi'])
map_T1._update_mask(map_DWI)
map_T2._update_mask(map_DWI)
map_T2.show_calc_stats_LV()
map_T1.show_calc_stats_LV()
map_DWI.show_calc_stats_LV()
map_T1_post._update_mask(map_DWI)
map_T1_post.show_calc_stats_LV()
#%%
def testing_plot(obj1,obj2,obj3, sl):
    
    %matplotlib inline
    alpha = 1.0*obj1.mask_lv[..., sl]

    print(f"Slice {sl}")
    # map map and overlay
    figsize = (4, 2)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, constrained_layout=True)

    for ind,obj in enumerate([obj1,obj2,obj3]):
        # map
        crange=obj.crange

        axes[ind,0].set_axis_off()
        im = axes[ind,0].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

        # map overlay
        base_sum=np.array([obj1._data[:, :, sl, 1],obj2._data[:, :, sl, 0],obj3._data[:, :, sl, 0]])
        base_im = np.mean(base_sum,axis=0)
        brightness = 0.8
        axes[ind,1].set_axis_off()
        axes[ind,1].imshow(base_im,vmax=np.max(base_im)*brightness, cmap="gray")
        im = axes[ind,1].imshow(obj._map[..., sl], alpha=alpha, vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

    plt.show()     
    pass
def testing_reseg(obj1,obj2,obj3,plot=plot):
    numberSlice=obj1.Nz
    obj=obj3
    for sl in range(numberSlice):
        testing_plot(obj1,obj2,obj3,sl)
        resegment = True
        
        # while resegment:
        # decide if resegmentation is needed
        print("Perform resegmentation? (Y/N)")
        tmp = input()
        resegment = (tmp == "Y") or (tmp == "y")
        
        while resegment:
            
            print("Resegment endo? (Y/N)")
            tmp = input()
            reseg_endo = (tmp == "Y") or (tmp == "y")
            
            print("Resegment epi? (Y/N)")
            tmp = input()
            reseg_epi = (tmp == "Y") or (tmp == "y") 
            
            roi_names = np.array(["endo", "epi"])
            roi_names = roi_names[np.argwhere([reseg_endo, reseg_epi]).ravel()]
            
            %matplotlib qt  
            # selectively resegment LV
            print("Kernel size: ")
            kernel = int(input())
            obj.go_resegment_LV(z=sl, roi_names=roi_names, dilate=True, kernel=kernel,image_type="b0_aveg")
            
            # re-plot
            testing_plot(obj1,obj2,obj3,sl)

            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            obj1._update_mask(obj)
            obj2._update_mask(obj)
            obj3._update_mask(obj)
            testing_plot(obj1,obj2,obj3,sl)
            obj1.show_calc_stats_LV()
            obj2.show_calc_stats_LV()
            obj3.show_calc_stats_LV()
    if plot:
        obj1.save(filename=os.path.join(img_save_dir,f'{obj1.CIRC_ID}_{obj1.ID}_p.mapping')) 
        obj2.save(filename=os.path.join(img_save_dir,f'{obj2.CIRC_ID}_{obj2.ID}_p.mapping')) 
        obj3.save(filename=os.path.join(img_save_dir,f'{obj3.CIRC_ID}_{obj3.ID}_p.mapping')) 
    pass
#%%
%matplotlib inline
testing_reseg(map_T1,map_T2,map_DWI)
# %% View Maps Overlay

%matplotlib inline
brightness=1.4
# view HAT mask
num_slice = map_T1.Nz 
figsize = (3.4*num_slice, 3)

# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_T1.crange
cmap=map_T1.cmap
base_sum=np.concatenate((map_T2._data[:, :, :, 0:5],map_DWI._data[:, :, :, 0:5]),axis=-1)
base_im = np.mean(base_sum,axis=-1)

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_T1._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_T1.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_T1.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_T2.crange
cmap=map_T2.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_T2._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_T2.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_T2.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_DWI.crange
cmap=map_DWI.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_DWI._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_DWI.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_DWI.ID}_overlay_maps.png"))
plt.show()  


# T1_post
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_T1_post.crange
cmap=map_T1_post.cmap
base_sum=np.concatenate((map_T2._data[:, :, :, 0:5],map_DWI._data[:, :, :, 0:5]),axis=-1)
base_im = np.mean(base_sum,axis=-1)

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_T1_post._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_T1_post.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_T1_post.ID}_overlay_maps.png"))
plt.show()  
plt.close()
#%%
map_T1.show_calc_stats_LV()
map_T2.show_calc_stats_LV()
map_DWI.show_calc_stats_LV()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3])


# %%
map_T1.save(filename=os.path.join(img_save_dir,f'{map_T1.CIRC_ID}_{map_T1.ID}_p_cropped.mapping'))
map_T2.save(filename=os.path.join(img_save_dir,f'{map_T2.CIRC_ID}_{map_T2.ID}_p_cropped.mapping'))
map_DWI.save(filename=os.path.join(img_save_dir,f'{map_DWI.CIRC_ID}_{map_DWI.ID}_p_cropped.mapping'))

# %%
