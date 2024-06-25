#%%
import argparse
import sys
from libMapping_v13 import mapping,readFolder  # <--- this is all you need to do diffusion processing
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
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Dec_14_2023')

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
#CIRC_ID_List=['446','452','429','419','407','405','398','382','381','373']
CIRC_ID_List=[446,452,429,419,405,398,382,381,373,457,472,486,498,500]
CIRC_NUMBER=CIRC_ID_List[8]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
#img_save_dir=os.path.join(img_root_dir,CIRC_ID)
img_save_dir = os.path.join(defaultPath, "saved_ims_v2_Dec_14_2023")

if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 

#%%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
data_path=os.path.dirname(dicomPath)
T1_bssfp,_,_  = readFolder(os.path.join(data_path,r'MR t1map_long_t1_3slice_8mm_150_gap_MOCO_T1-2'))

T2_bssfp,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_3slice_8mm_150_gap_MOCO_T2'))

#%%
from imgbasics import imcrop
from skimage.transform import resize as imresize
%matplotlib qt

data=T2_bssfp.squeeze()
map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)

map.shape=np.shape(map._data)
map.go_crop()
#map.go_resize(scale=2)
map.shape=np.shape(map._data)
cropzone=map.cropzone
#%%
#################################Second Run:###############################
###########################Please Copy to the Console#####################
#data=T2_bssfp.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)


#data=T1_bssfp_fb.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI_FB',CIRC_ID=CIRC_ID)
#map.shape=np.shape(map._data)
#map.go_crop()

#map.shape=np.shape(map._data)
#cropzone=map.cropzone

data=T1_bssfp.squeeze()
map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI',CIRC_ID=CIRC_ID)

#%%
map.cropzone=cropzone
map.shape=np.shape(map._data)
map.go_crop()
#map.go_resize(scale=2)
map.shape=np.shape(map._data)
#%%
##############################T1#################################
%matplotlib qt
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map._data))
crange=[0,3000]
map.crange=crange
map._map=data_crop.squeeze()
map.cropzone=cropzone
print(map.shape)
map.path=dicomPath
map.cmap="magma"
map.go_segment_LV(image_type='map',crange=crange,cmap="magma")
# %%
##############################T2#################################
#Truncation and Resize and MOCO
%matplotlib qt
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
map.path=dicomPath
map.cmap="viridis"
map.go_segment_LV(image_type='map',crange=crange,cmap="viridis")

#%%
%matplotlib inline

map.show_calc_stats_LV()
testing_reseg(map)


# %%

%matplotlib inline

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
#map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Dec.csv',crange=crange)
map.save(filename=os.path.join(img_save_dir,f'{map.CIRC_ID}_{map.ID}_p.mapping'))
print('Saved the segmentation sucessfully')
#Save again
#map.save()
print('Saved the segmentation sucessfully')
map.imshow_overlay(path=img_save_dir)

# %%
