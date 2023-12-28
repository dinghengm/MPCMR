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
CIRC_ID_List=['446','452','429','419','407','405','398','382','381','373']
CIRC_NUMBER=CIRC_ID_List[9]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Dec_14_2023')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            if CIRC_ID in path:
                mapList.append(path)
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[5])
#%%
%matplotlib inline
img_save_dir=os.path.join(img_root_dir,CIRC_ID)
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
MP02.imshow_map(plot=False,path=img_save_dir)
MP01.imshow_map(plot=False,path=img_save_dir)
MP03.imshow_map(plot=False,path=img_save_dir)
#%%
%matplotlib qt
MP02.go_segment_LV(reject=None, image_type="b0_avg")

#%%
MP02.show_calc_stats_LV()
MP03._update_mask(MP02)
MP03.show_calc_stats_LV()

MP01._update_mask(MP02)
MP01.show_calc_stats_LV()

#%%
def testing_plot(obj, sl):
    
    
    alpha = 1.0*obj.mask_lv[..., sl]

    print(f"Slice {sl}")
    # map map and overlay
    figsize = (4, 2)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, constrained_layout=True)

    # map
    crange=obj.crange

    axes[0].set_axis_off()
    im = axes[0].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

    # map overlay
    base_im = obj._data[:, :, sl, 0]
    brightness = 0.8
    axes[1].set_axis_off()
    axes[1].imshow(base_im,vmax=np.max(base_im)*brightness, cmap="gray")
    im = axes[1].imshow(obj._map[..., sl], alpha=alpha, vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

    plt.show()     
def testing_reseg(obj):
    numberSlice=obj.Nz
    for sl in range(numberSlice):
        testing_plot(obj,sl)
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
            obj.go_resegment_LV(z=sl, roi_names=roi_names, dilate=True, kernel=kernel,image_type="map")
            
            # re-plot
            testing_plot(obj, sl) 

            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            obj.show_calc_stats_LV()
    obj.save(filename=os.path.join(img_root_dir,f'{obj.CIRC_ID}_{obj.ID}_p.mapping')) 

#%%
testing_reseg(MP02)

#%%
testing_reseg(MP03)
#%%
testing_reseg(MP01)

# %% VIEW MaPS
%matplotlib inline

# create images images per map type
num_slice = MP01.Nz
figsize = (3.4*num_slice, 3)

# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP01.crange
cmap=MP01.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
plt.savefig(os.path.join(img_save_dir, f"{MP01.ID}"))
plt.show()

#T2 
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP02.crange
cmap=MP02.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
plt.savefig(os.path.join(img_save_dir, f"{MP02.ID}"))
plt.show()

#ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP03.crange
cmap=MP03.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
plt.savefig(os.path.join(img_save_dir, f"{MP03.ID}"))
plt.show()

# %% View Maps Overlay

%matplotlib inline

# view HAT mask
num_slice = MP01.Nz 
figsize = (3.4*num_slice, 3)

# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP01.crange
cmap=MP01.cmap
base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(os.path.join(img_save_dir, f"{MP01.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP02.crange
cmap=MP02.cmap
base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(os.path.join(img_save_dir, f"{MP02.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP03.crange
cmap=MP03.cmap
base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(os.path.join(img_save_dir, f"{MP03.ID}_overlay_maps.png"))
plt.show()  



#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Dec.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Dec.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Dec.csv',crange=[0,3])


# %%
MP01.save(filename=os.path.join(img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))
MP02.save(filename=os.path.join(img_root_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_root_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))

# %%
