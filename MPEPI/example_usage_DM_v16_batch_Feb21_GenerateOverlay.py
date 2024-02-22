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
import warnings #we know deprecation may show bc we are using a stable older ITK version
defaultPath= r'C:\Research\MRI\MP_EPI'
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
#%%

CIRC_ID_List=[446,429,419,405,398,382,381,373,472,486,498,500]
#CIRC_NUMBER=CIRC_ID_List[9]
for ind in range(len(CIRC_ID_List))[-3:-1]:
    CIRC_NUMBER=CIRC_ID_List[ind]

    CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
    print(f'Running{CIRC_ID}')
    img_root_dir=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737')
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('MOLLI_p.mapping') or path.endswith('FLASH_p.mapping'):
                    if 'FB' not in path:
                        mapList.append(path)
    #T1_FB=mapping(mapList[-4])
    T1=mapping(mapList[-2])
    #T2_FB=mapping(mapList[-2])
    T2=mapping(mapList[-1])
    print(mapList)
    img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('p.mapping'):
                mapList.append(path)
    MP01=mapping(mapList[0])
    MP02=mapping(mapList[1])
    #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
    #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
    MP03=mapping(mapList[2])


    def imshowMap(obj,path,plot):
        num_slice=obj.Nz
        volume=obj._map
        ID=str('map_' + obj.CIRC_ID + '_' + obj.ID)
        if 'T1' in ID:
            crange=[0,3000]
            cmap='magma'
        if 'T2' in ID:
            crange=[0,150]
            cmap='viridis'
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
    maps_save_dir=os.path.join(os.path.dirname(img_root_dir),'Clinical_maps')
    %matplotlib inline
    if not os.path.exists(maps_save_dir):
        os.makedirs(maps_save_dir) 
    #imshowMap(obj=MP02,plot=plot,path=maps_save_dir)
    #imshowMap(obj=MP01,plot=plot,path=maps_save_dir)
    #imshowMap(obj=MP03,plot=plot,path=maps_save_dir)
    try:
        imshowMap(obj=T1,plot=plot,path=maps_save_dir)
        imshowMap(obj=T2,plot=plot,path=maps_save_dir)
        #maps_save_dir=os.path.join(os.path.dirname(img_root_dir),'Clinical_maps','FB')
        %matplotlib inline
        #if not os.path.exists(maps_save_dir):
        #    os.makedirs(maps_save_dir) 
        #imshowMap(obj=T2_FB,plot=plot,path=maps_save_dir)
        #imshowMap(obj=T1_FB,plot=plot,path=maps_save_dir)
    except:
        pass
    def testing_plot(obj1,obj2,obj3, sl):

        %matplotlib inline
        alpha = 1.0*obj1.mask_lv[..., sl]

        print(f"Slice {sl}")
        # map map and overlay
        figsize = (6, 12)
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
        maps_save_dir=os.path.join(os.path.dirname(img_root_dir),'singleSlice','MP-EPI')
        if plot:
            plt.savefig(os.path.join(maps_save_dir,f'{obj.CIRC_ID}_{sl}'))
        plt.show()  
        pass


    %matplotlib inline
    for ss in range(3):
        testing_plot(MP01,MP02,MP03,ss)

    %matplotlib inline
    overlay_save_dir=os.path.join(os.path.dirname(img_root_dir),'singleSlice','Overlay')
    num_slice = MP01.Nz 
    figsize = (3.4*num_slice, 3*3)

    # T1
    fig, axes = plt.subplots(nrows=3, ncols=num_slice, figsize=figsize, constrained_layout=True)


    cmap='magma'

    crange=[0,3000]
    base_sum=np.array([MP02._data[:, :, :, 0:6],MP03._data[:, :, :, 0:6]])
    base_im = np.mean(base_sum,axis=0)
    base_im = np.mean(base_im,axis=-1)
    for sl in range(num_slice):
        axes[sl,0].set_axis_off()
        axes[sl,0].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
        axes[sl,0].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
    #plt.show()  
    #plt.close()
    # T2


    cmap='viridis'

    crange=[0,150]

    #base_im = MP03._data[..., 0]

    for sl in range(num_slice):
        axes[sl,1].set_axis_off()
        axes[sl,1].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
        axes[sl,1].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
    #plt.show()  
    #plt.close()
    # ADC
    
    
    
    crange=MP03.crange
    cmap=MP03.cmap


    
    #base_im = MP03._data[..., 0]

    for sl in range(num_slice):
        axes[sl,2].set_axis_off()
        axes[sl,2].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
        im = axes[sl,2].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
    if plot:
        plt.savefig(os.path.join(overlay_save_dir, f"{MP03.CIRC_ID}_overlay_maps.png"))
    plt.show()  

    
    #Plot the T1/T2 side by side

    %matplotlib inline
    figsize = (3.4*2, 3*num_slice)

    # map map and overlay

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, constrained_layout=True)
    for sl in range(num_slice):
        axes[sl,0].set_axis_off()
        im = axes[sl,0].imshow(T1._map[..., sl], vmin=0, vmax=3000, cmap='magma')

    for sl in range(num_slice):
        axes[sl,1].set_axis_off()
        im = axes[sl,1].imshow(T2._map[..., sl], vmin=0, vmax=150, cmap='viridis')

    maps_save_dir=os.path.join(os.path.dirname(img_root_dir),'singleSlice','Clinical_maps')
    if plot:
        plt.savefig(os.path.join(maps_save_dir, f"{T1.CIRC_ID}_Clinical_maps.png"))
    plt.show()     


# %%
%matplotlib inline
overlay_save_dir=os.path.join(os.path.dirname(img_root_dir),'singleSlice','Overlay')
num_slice = MP01.Nz 
figsize = (3.4*2, 3*num_slice)

# T1
fig, axes = plt.subplots(nrows=3, ncols=num_slice, figsize=figsize, constrained_layout=True)
norm = matplotlib.colors.Normalize(vmin=0, vmax=3000) 

cmap='magma'
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm) 
crange=[0,3000]
base_sum=np.array([MP02._data[:, :, :, 0:6],MP03._data[:, :, :, 0:6]])
base_im = np.mean(base_sum,axis=0)
base_im = np.mean(base_im,axis=-1)
for sl in range(num_slice):
    axes[sl,0].set_axis_off()
    axes[sl,0].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    axes[sl,0].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
#plt.show()  
#plt.close()
# T2
cbar = fig.colorbar(sm,  aspect=10, orientation='horizontal', ax=axes[2,0])


norm = matplotlib.colors.Normalize(vmin=0, vmax=150) 

cmap='viridis'
sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm) 
crange=[0,150]

#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl,1].set_axis_off()
    axes[sl,1].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    axes[sl,1].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
#plt.show()  
#plt.close()
# ADC
cbar = fig.colorbar(sm,  aspect=10, orientation='horizontal', ax=axes[2,1])



crange=MP03.crange
cmap=MP03.cmap
norm = matplotlib.colors.Normalize(vmin=0, vmax=3) 

sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm) 
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl,2].set_axis_off()
    axes[sl,2].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl,2].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)


cbar = fig.colorbar(sm,  aspect=10, orientation='horizontal', ax=axes[2,2])
plt.show()  