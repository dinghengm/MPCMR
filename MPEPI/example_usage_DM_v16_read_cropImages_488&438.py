# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline                     
from libMapping_v13 import *  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
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
CIRC_ID='CIRC_00438'
img_root_dir = 'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00438'
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'CIRC_00438')
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
        if path.endswith('p.mapping'):
            
            #if not path.endswith('m.mapping') and not path.endswith('p.mapping'):
                mapList.append(path)
print(mapList)

#%%

MP01=mapping(mapList[0])
MP02=mapping(mapList[1])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[2])
#%%
#Get the first slice from else where

MP01_temp=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Dec_14_2023\CIRC_00438\MP01_T1_p.mapping')
#%%
MP01._map[...,:]=MP01_temp._map[...,:]


#%%
#Crop it a little smaller
%matplotlib qt
MP02.go_crop()
cropzone=MP02.cropzone
#%%
MP01.cropzone=cropzone
MP03.cropzone=cropzone
MP01.go_crop()
MP03.go_crop()


#%%
%matplotlib qt 

#%%
plt.style.use('default')
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

#img_save_dir=os.path.join(img_root_dir,CIRC_ID)
img_save_dir=saved_img_root_dir
%matplotlib inline
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
imshowMap(obj=MP02,plot=plot,path=img_save_dir)
imshowMap(obj=MP01,plot=plot,path=img_save_dir)
imshowMap(obj=MP03,plot=plot,path=img_save_dir)


#%%
%matplotlib qt
MP02.go_segment_LV(reject=None, image_type="b0_avg",roi_names=['endo', 'epi'])

#%%
MP02.show_calc_stats_LV()
MP03._update_mask(MP02)
MP03.show_calc_stats_LV()

MP01._update_mask(MP02)
MP01.show_calc_stats_LV()

#MP01_post._update_mask(MP02)
#MP01_post.show_calc_stats_LV()
#%%
def testing_plot(obj1,obj2,obj3,obj4,sl):
    
    %matplotlib inline
    alpha = 1.0*obj1.mask_lv[..., sl]

    print(f"Slice {sl}")
    # map map and overlay
    figsize = (4, 2)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize, constrained_layout=True)

    for ind,obj in enumerate([obj1,obj2,obj3,obj4]):
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
def testing_reseg(obj1,obj2,obj3,obj4):
    numberSlice=obj1.Nz
    obj=obj2
    for sl in range(numberSlice):
        testing_plot(obj1,obj2,obj3,obj4,sl)
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
            testing_plot(obj1,obj2,obj3,obj4,sl)

            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            obj1._update_mask(obj2)
            obj3._update_mask(obj2)
            obj4._update_mask(obj2)
            testing_plot(obj1,obj2,obj3,obj4,sl)
            obj1.show_calc_stats_LV()
            obj2.show_calc_stats_LV()
            obj3.show_calc_stats_LV()
            obj4.show_calc_stats_LV()

    pass
#%%
%matplotlib inline
plt.style.use('default')
testing_reseg(MP01,MP03,MP02,MP03)


#%%
plt.style.use('default')

img_save_dir=saved_img_root_dir
%matplotlib inline
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
MP01._update_mask(MP02)
MP03._update_mask(MP02)
MP01.show_calc_stats_LV()
# %% View Maps Overlay

%matplotlib inline
img_save_dir=saved_img_root_dir

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
    plt.savefig(os.path.join(img_save_dir, f"{MP01.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
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
    plt.savefig(os.path.join(saved_img_root_dir, f"MP01_Slice2_try4_overlay_maps.png"))
plt.show()  
plt.close()

MP01.show_calc_stats_LV()
#MP01_post.show_calc_stats_LV()

# T2
#MP03.go_cal_ADC()
#MP02.go_t2_fit()

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
MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))

MP02.save(filename=os.path.join(saved_img_root_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(saved_img_root_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))

# %%
