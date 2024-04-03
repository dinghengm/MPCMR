#%%
'''
    This file read the p.mapping from the 'MP-EPI-CIRC_ID_IRB'
    And then can re-adjust the clinical reference maps from the data: T2-FLASH AND T1-MOLLI
    Save the file into MPEPI-Feb_5-Reference
    Save the data into MP-EPI-Feb_Referemne
    This might be the final step on getting the reference values

'''

import argparse
import sys
from libMapping_v13 import mapping,readFolder  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
defaultPath= r'C:\Research\MRI\MP_EPI'
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib

matplotlib.rcParams['savefig.dpi'] = 400
plot=True


#%%
#CIRC_ID_List=['446','452','429','419','405','398','382','381','373']
CIRC_ID_List=[446,429,419,405,398,382,381,373,472,486,498,500]
CIRC_NUMBER=CIRC_ID_List[7]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
CIRC_ID_IRB=f'CIRC_00{CIRC_NUMBER}_22737_CIRC_00{CIRC_NUMBER}_22737'
print(f'Running{CIRC_ID_IRB}')
global img_save_dir
img_root_dir=os.path.join(defaultPath,CIRC_ID_IRB)
img_save_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','Reference')
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('T1_MOLLI_p.mapping') or path.endswith('T2_FLASH_p.mapping'):
            if 'FB' not in path:
                mapList.append(path)
print(mapList)
T1=mapping(mapList[0])
T2=mapping(mapList[1])

#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
#%%
def testing_plot(obj, sl):
    
    %matplotlib inline
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
            obj.show_calc_stats_LV()
            testing_plot(obj, sl) 
            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            
            
    obj.save(filename=os.path.join(img_save_dir,f'{obj.CIRC_ID}_{obj.ID}_p.mapping')) 

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
T1.show_calc_stats_LV()
T2.show_calc_stats_LV()
maps_save_dir=os.path.join(img_save_dir,'Maps')
imshowMap(obj=T1,plot=plot,path=maps_save_dir)
imshowMap(obj=T2,plot=plot,path=maps_save_dir)
for obj in [T1,T2]:
    for sl in range(3):
        testing_plot(obj, sl)




# %%
%matplotlib inline
for obj in [T1]:
    testing_reseg(obj)
    #View Maps Overlay
    
    %matplotlib inline

    # view HAT mask
    num_slice = obj.Nz 
    figsize = (3.4*num_slice, 3)

    # T1
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=obj.crange
    cmap=obj.cmap
    base_im=obj._data[...,0]
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
        
        im = axes[sl].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*obj.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
    plt.savefig(os.path.join(img_save_dir, 'Overlay',f"{obj.CIRC_ID}_{obj.ID}_overlay_maps.png"))
    plt.show()  
    plt.close()



# %%
%matplotlib inline
for obj in [T2]:
    testing_reseg(obj)
    #View Maps Overlay
    
    %matplotlib inline

    # view HAT mask
    num_slice = obj.Nz 
    figsize = (3.4*num_slice, 3)

    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=obj.crange
    cmap=obj.cmap
    base_im=obj._data[...,0]
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
        
        im = axes[sl].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*obj.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
    plt.savefig(os.path.join(img_save_dir, 'Overlay',f"{obj.CIRC_ID}_{obj.ID}_overlay_maps.png"))
    plt.show()  
    plt.close()

#%%
T1.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_April_Reference.csv',crange=[0,3000])
T2.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_April_Reference.csv',crange=[0,150])


# %%
for obj in [T1,T2]:
    obj.save(filename=os.path.join(img_save_dir,f'{obj.CIRC_ID}_{obj.ID}_p.mapping'))

# %%

#%%
#########Optional
T1.go_resize()
T1._map=np.copy(T1._data).squeeze()
T2.go_resize()
T2._map=np.copy(T2._data).squeeze()/10
###########
#%%
%matplotlib qt
T1.go_segment_LV()
T2.go_segment_LV()

#%%
T1.cmap='magma'
T2.cmap='viridis'
T2.crange=[0,150]
T1.crange=[0,3000]