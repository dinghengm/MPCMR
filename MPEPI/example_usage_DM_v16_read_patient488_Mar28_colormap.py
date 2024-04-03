# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline                     
from libMapping_v13 import *  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
#%%
boolenTest=input('Would you like to save you Plot? "Y" and "y" to save')
if boolenTest.lower() == 'y':
    plot=True
else:
    plot=False


# %%
#Please try to change to CIRC_ID
CIRC_ID='CIRC_00488'
img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'CIRC_00488_1')
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'CIRC_00488_1')
if not os.path.exists(saved_img_root_dir):
    os.mkdir(saved_img_root_dir)

#img_root_dir = saved_img_root_dir
# image root directory
# Statas saved 
stats_file = os.path.join(defaultPath, "MPEPI_stats_v16.csv") 


#Read the MP01-MP03
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            
            #if not path.endswith('m.mapping') and not path.endswith('p.mapping'):
                mapList.append(path)
print(mapList)

#%%
MP01_post_0=mapping(mapList[0])
MP01_post_1=mapping(mapList[1])
MP01_post_2=mapping(mapList[2])
MP01_0=mapping(mapList[3])
MP01_1=mapping(mapList[4])
MP01_2=mapping(mapList[5])
MP01=mapping(mapList[6])
MP01_post=mapping(mapList[7])
MP02=mapping(mapList[8])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[9])


#%%

MP01_2try_1=mapping(os.path.join(saved_img_root_dir,'T1_2_m.mapping'))

#%%

del MP01
MP01=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488\CIRC_00488_MP01_T1_p.mapping')
MP02_contour=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488\CIRC_00488_MP02_T2_p.mapping')
for obj in [MP01]:
    obj.go_crop_Auto()
    obj.go_resize()
    obj._update()
#%%
plt.style.use('default')

MP01._map[:,:,2]=np.squeeze(MP01_2try_1._map)
#img_save_dir=os.path.join(img_root_dir,CIRC_ID)
img_save_dir=saved_img_root_dir
%matplotlib inline
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
MP01._update_mask(MP02_contour)
MP01.show_calc_stats_LV()
#%%
#View Maps Overlay
%matplotlib inline
# view HAT mask
num_slice = MP01.Nz 
figsize = (3.4*num_slice, 3)
# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP01.crange
cmap=MP01.cmap
#base_sum=np.array([MP02._data[:, :, :, 0:6],MP03._data[:, :, :, 0:6]])
#base_im = np.mean(base_sum,axis=0)
base_sum=np.concatenate([MP02._data[:, :, :, 0:6],MP03._data[:, :, :,  0:12]],axis=-1)
base_im = np.mean(base_sum,axis=-1)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
    im = axes[sl].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(saved_img_root_dir, f"MP01_Slice2_try{ss}_overlay_maps.png"))
plt.show()  
plt.close()

MP01.show_calc_stats_LV()
#MP01_post.show_calc_stats_LV()
MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))

# %%
# T2
#MP03.go_cal_ADC()
#MP02.go_t2_fit()
MP02._update_mask(MP02_contour)
MP03._update_mask(MP02_contour)
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP02.crange
cmap=MP02.cmap
#ase_im = MP02._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
    im = axes[sl].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{MP02.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP03.crange
cmap=MP03.cmap
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
    im = axes[sl].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{MP03.ID}_overlay_maps.png"))
plt.show()  

# %%
MP02.save(filename=os.path.join(saved_img_root_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(saved_img_root_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))

# %%
