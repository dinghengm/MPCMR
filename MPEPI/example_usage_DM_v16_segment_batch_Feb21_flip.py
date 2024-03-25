###############Batch contour for paper
##############From  saved_ims_v2_Feb_5_2024/WITH8000
##############TO  saved_ims_v2_Feb_5_2024/WITH8000/Overlay
##############Please draw on DWI
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
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)
MP01=mapping(mapList[-3])
MP02=mapping(mapList[-2])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[-1])
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
#Flip the data#####################################################
def flipData(obj):
    obj._data=np.flip(obj._data,0)
    obj._raw_data=np.flip(obj._raw_data,0)
    obj._map=np.flip(obj._map,0)
    obj.mask_endo=np.flip(obj.mask_endo,0)
    obj.mask_epi = np.flip(obj.mask_epi,0)
    obj.mask_lv =np.flip(obj.mask_lv,0)
    try:
        obj.mask_septal =np.flip(obj.mask_septal,0)
        obj.mask_lateral = np.flip(obj.mask_lateral,0)
    except:
        pass

    obj._update()

flipData(MP02)
flipData(MP03)
flipData(MP01)

#%%
def cropData(obj,cropStartVx=32):
    obj._data=obj._crop_Auto(obj._data,cropStartVx)
    obj._map=obj._crop_Auto(obj._map[:,:,:,np.newaxis],cropStartVx).squeeze()
    obj.mask_endo=obj._crop_Auto(obj.mask_endo[:,:,:,np.newaxis],cropStartVx).squeeze()
    obj.mask_epi =obj._crop_Auto(obj.mask_epi[:,:,:,np.newaxis],cropStartVx).squeeze()
    obj.mask_lv =obj._crop_Auto(obj.mask_lv[:,:,:,np.newaxis],cropStartVx).squeeze()
    try:
        obj.mask_septal =obj._crop_Auto(obj.mask_septal[:,:,:,np.newaxis],cropStartVx).squeeze()
        obj.mask_lateral = obj._crop_Auto(obj.mask_lateral[:,:,:,np.newaxis],cropStartVx).squeeze()
    except:
        pass

    obj._update()

cropData(MP02,64)
cropData(MP03,64)
cropData(MP01,64)

#%%



maps_save_dir=os.path.join(os.path.dirname(img_root_dir),'maps')
img_save_dir=img_root_dir

%matplotlib inline
if not os.path.exists(maps_save_dir):
    os.makedirs(maps_save_dir) 
imshowMap(obj=MP02,plot=plot,path=maps_save_dir)
imshowMap(obj=MP01,plot=plot,path=maps_save_dir)
imshowMap(obj=MP03,plot=plot,path=maps_save_dir)


#%%
%matplotlib qt
MP03.go_segment_LV(reject=None, image_type="b0",roi_names=['endo', 'epi','septal','lateral'],crange=[0,150])
#MP02.go_segment_LV(reject=None, image_type="map",roi_names=['endo', 'epi','septal','lateral'])

#%%

MP02._update_mask(MP03)


MP01._update_mask(MP02)
MP02.show_calc_stats_LV()
MP01.show_calc_stats_LV()
MP03.show_calc_stats_LV()
#%%
def testing_plot(obj1,obj2,obj3, sl):
    
    %matplotlib inline
    alpha = 1.0*obj1.mask_lv[..., sl]

    print(f"Slice {sl}")
    # map map and overlay
    figsize = (16, 4)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, constrained_layout=True)

    for ind,obj in enumerate([obj1,obj2,obj3]):
        # map
        crange=obj.crange

        axes[ind,0].set_axis_off()
        im = axes[ind,0].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

        # map overlay
        base_sum=np.array([obj2._data[:, :, sl, 0],obj3._data[:, :, sl, 0]])
        base_im = np.mean(base_sum,axis=0)
        brightness = 0.8
        axes[ind,1].set_axis_off()
        axes[ind,1].imshow(base_im,vmax=np.max(base_im)*brightness, cmap="gray")
        im = axes[ind,1].imshow(obj._map[..., sl], alpha=alpha, vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

    plt.show()     
    pass
def testing_reseg(obj1,obj2,obj3):
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
            obj.go_resegment_LV(z=sl, roi_names=roi_names, dilate=True, kernel=kernel,image_type="map")
            
            # re-plot
            testing_plot(obj1,obj2,obj3,sl)

            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            obj1._update_mask(obj)
            obj3._update_mask(obj)
            obj2._update_mask(obj)
            testing_plot(obj1,obj2,obj3,sl)
            obj1.show_calc_stats_LV()
            obj2.show_calc_stats_LV()
            obj3.show_calc_stats_LV()
    obj1.save(filename=os.path.join(img_save_dir,f'{obj1.CIRC_ID}_{obj1.ID}_p.mapping')) 
    obj2.save(filename=os.path.join(img_save_dir,f'{obj2.CIRC_ID}_{obj2.ID}_p.mapping')) 
    obj3.save(filename=os.path.join(img_save_dir,f'{obj3.CIRC_ID}_{obj3.ID}_p.mapping')) 
    pass

#%%
%matplotlib inline
testing_reseg(MP01,MP02,MP03)

# %% View Maps Overlay

%matplotlib inline

overlay_save_dir=os.path.join(os.path.dirname(img_root_dir),'overlay')

num_slice = MP01.Nz 
figsize = (3.4*num_slice, 3)

# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP01.crange
cmap=MP01.cmap
base_sum=np.array([MP02._data[:, :, :, 0:6],MP03._data[:, :, :, 0:6]])
base_im = np.mean(base_sum,axis=0)
base_im = np.mean(base_im,axis=-1)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(overlay_save_dir, f"{MP01.CIRC_ID}_{MP01.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP02.crange
cmap=MP02.cmap
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(overlay_save_dir, f"{MP02.CIRC_ID}_{MP02.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP03.crange
cmap=MP03.cmap
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(overlay_save_dir, f"{MP03.CIRC_ID}_{MP03.ID}_overlay_maps.png"))
plt.show()  


#%%
MP02.show_calc_stats_LV()
MP03.show_calc_stats_LV()
MP01.show_calc_stats_LV()
#%%
filename=os.path.join(os.path.dirname(img_root_dir),'mapping_Feb_Ori.csv')
MP01.export_stats(filename=filename,crange=[0,1800])
MP02.export_stats(filename=filename,crange=[0,150])
MP03.export_stats(filename=filename,crange=[0,3])


# %%
MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))
MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))

# %%

########################Use the maps before
MP03._data=MP03._raw_data
MP03.go_crop_Auto()
MP03.go_resize()
MP03.go_cal_ADC()
datatmp=MP02._data[:,:,:,0]
data=np.concatenate((datatmp[:,:,:,np.newaxis],MP03._data),axis=-1)
from libMapping_v13 import moco
moco_data=moco(data,MP02)
MP03._data=np.delete(moco_data,0,axis=-1)

MP03.go_cal_ADC()
# %%
