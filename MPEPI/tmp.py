#%%
####################Currently I am testing if I could get click on the center of mass and also the edge of 
####################
from libMapping_v14 import *  # <--- this is all you need to do diffusion processing
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
#Get the center
%matplotlib qt
map_T1.go_define_CoMandRVIns()

#%%
#Show the center in the first slice
plt.figure()
cmap=map_T1.cmap
crange=map_T1.crange
plt.imshow(map_T1._map[:,:,0].squeeze(),cmap='gray')

CoMTmp=map_T1.CoM[0]
aRVIns=map_T1.aRVIns[0]
iRVIns=map_T1.iRVIns[0]
#So stupid!!! What the fuck!!!
# Define points
anterior_rv_insertion = np.array([aRVIns[1], aRVIns[0]])  # example coordinates
inferior_rv_insertion = np.array([iRVIns[1], iRVIns[0]])  # example coordinates
center_of_mass = np.array([CoMTmp[1], CoMTmp[0]])  # example coordinates (center)



plt.plot(*center_of_mass,'ro')
plt.plot(*inferior_rv_insertion,'ro')
plt.plot(*anterior_rv_insertion,'ro')

#plt.plot(CoMTmp[1],CoMTmp[0],'ro')
#plt.plot(aRVIns[1],aRVIns[0],'ro')
#plt.plot(iRVIns[1],iRVIns[0],'ro')
#plt.xlim(0,np.shape(map_T1._map)[1])
#plt.ylim(0,np.shape(map_T1._map)[0])
plt.show()

#%%
#Show the insertion of the wheel 
plt.figure()
plt.imshow(map_T1.mask_lv[:,:,0].squeeze())

midpoint = (anterior_rv_insertion + inferior_rv_insertion) / 2
plt.plot(*center_of_mass,'ro')
plt.plot(*inferior_rv_insertion,'ro')
plt.plot(*anterior_rv_insertion,'ro')
#The line not the angle
plt.plot([center_of_mass[0], anterior_rv_insertion[0]], [center_of_mass[1], anterior_rv_insertion[1]], 'k--')
plt.plot([center_of_mass[0], inferior_rv_insertion[0]], [center_of_mass[1], inferior_rv_insertion[1]], 'k--')
plt.plot([center_of_mass[0], midpoint[0]], [center_of_mass[1], midpoint[1]], 'k--')


#%%
# Calculate angles for segment lines
plt.figure()
plt.imshow(map_T1.mask_lv[:,:,0].squeeze())
plt.plot(*center_of_mass,'ro')
plt.plot(*inferior_rv_insertion,'ro')
plt.plot(*anterior_rv_insertion,'ro')
angle1 = np.arctan2(anterior_rv_insertion[1] - center_of_mass[1], anterior_rv_insertion[0] - center_of_mass[0])
angle2 = np.arctan2(inferior_rv_insertion[1] - center_of_mass[1], inferior_rv_insertion[0] - center_of_mass[0])
angle3 = (angle1 + angle2) / 2

# Define the six end point for the lines (end point is needed to make sure it covers the whole mask_LV)
# Also get the CoM
#Define the 3 lines based on the CoM and each end point
#First line: bisector
line_length = 40  # Adjust the length of the line, to make sure it's long enough to cover the mask_LV
bisector_start= center_of_mass - line_length * np.array([np.cos(angle3), np.sin(angle3)])
bisector_end = center_of_mass + line_length * np.array([np.cos(angle3), np.sin(angle3)])
#Second line: the anterior part
line_scale=1.5   #Random value to make sure it's long enough to cover the Mask_LV
anterior_start=center_of_mass - line_scale * (center_of_mass-anterior_rv_insertion)
anterior_end=center_of_mass + line_scale * (center_of_mass-anterior_rv_insertion)
#Thrid line: the inferior part
inferior_start=center_of_mass - line_scale * (center_of_mass-inferior_rv_insertion)
inferior_end=center_of_mass + line_scale * (center_of_mass-inferior_rv_insertion)

plt.plot([bisector_start[0], bisector_end[0]], [bisector_start[1], bisector_end[1]], 'k--')
plt.plot([anterior_start[0], anterior_end[0]], [anterior_start[1], anterior_end[1]], 'k--')
plt.plot([inferior_start[0], inferior_end[0]], [inferior_start[1], inferior_end[1]], 'k--')

#%%
from skimage.draw import polygon2mask
#Define the first coordinate:
Nx=map_T1.Nx
Ny=map_T1.Ny
coordinates = (bisector_start, center_of_mass, anterior_start) #3 points
#Sometimes it's reverted
coordinates = [[y,x] for [x,y] in coordinates]
polygon = np.array(coordinates)
mask = polygon2mask([Nx,Ny], polygon)
plt.figure()
mask2=np.logical_and(mask,map_T1.mask_lv[:,:,0])
plt.imshow(mask2)
plt.show()


#%%
#Find the overlap between three part

mask_lv_nn=map_T1.mask_lv[:,:,0]
def coordinate2mask(coordinates,mask_lv_nn,Nx,Ny):
            #for skimage it's reverted
            coordinates = [[y,x] for [x,y] in coordinates]
            polygon = np.array(coordinates)
            #Use the polygon2mask fuction for mask generation
            mask_tmp = polygon2mask([Nx,Ny], polygon)
            #Found the overlap between mask_seg and mask_lv_nn
            mask_seg=np.logical_and (mask_tmp, mask_lv_nn)
            return mask_seg

coordinates = (inferior_end, center_of_mass, anterior_start)
mask_seg=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny) *1

coordinates = (bisector_start, center_of_mass, anterior_start)
mask_seg2=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)*2

coordinates = (bisector_start, center_of_mass, inferior_start)
mask_seg3=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)*3

coordinates = (anterior_end, center_of_mass, inferior_start)
mask_seg4=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)*4

coordinates = (anterior_end, center_of_mass, bisector_end)
mask_seg5=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)*5

coordinates = (inferior_end, center_of_mass, bisector_end)
mask_seg6=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)*6

maskFinal=mask_seg+mask_seg2+mask_seg3+mask_seg4+mask_seg5+mask_seg6

plt.imshow(maskFinal,vmax=6)
plt.show()
#%%
segment_16=[]
mask_lv_nn=map_T1.mask_lv[:,:,2].squeeze()
#For the first slice:
CoMTmp=map_T1.CoM[2]
aRVIns=map_T1.aRVIns[2]
iRVIns=map_T1.iRVIns[2]
# Define points
#The x and y is reversed from previously 
anterior_rv_insertion = np.array([aRVIns[1], aRVIns[0]])  # example coordinates
inferior_rv_insertion = np.array([iRVIns[1], iRVIns[0]])  # example coordinates
center_of_mass = np.array([CoMTmp[1], CoMTmp[0]])  # example coordinates (center)

#Second line: the anterior part
line_scale=1.5   #Random value to make sure it's long enough to cover the Mask_LV
anterior_start=center_of_mass - line_scale * (center_of_mass-anterior_rv_insertion)
anterior_end=center_of_mass + line_scale * (center_of_mass-anterior_rv_insertion)
#Thrid line: the inferior part
inferior_start=center_of_mass - line_scale * (center_of_mass-inferior_rv_insertion)
inferior_end=center_of_mass + line_scale * (center_of_mass-inferior_rv_insertion)

#Number 13 Seg: Anterior 
coordinates = (inferior_end, center_of_mass, anterior_start)
mask_seg=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)
segment_16.append(mask_seg)

#Number 14 Seg: Septal 
coordinates = (inferior_start, center_of_mass, anterior_start)
mask_seg=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)
segment_16.append(mask_seg)

#Number 15 Seg:  Inferior
coordinates = (anterior_end, center_of_mass, inferior_start)
mask_seg=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)
segment_16.append(mask_seg)

#Number 16 Seg:  Lateral
coordinates = (anterior_end, center_of_mass, inferior_end)
mask_seg=coordinate2mask(coordinates,mask_lv_nn,Nx,Ny)
segment_16.append(mask_seg)
maskFinal=np.zeros((Nx,Ny),dtype=int)
for nn,masktmp in enumerate(segment_16):
    maskFinal+=segment_16[nn] *(nn+1)

plt.imshow(maskFinal,vmax=4)
plt.show()

#%%
#Copy the map from T1 T2 and ADC
map_T1.go_get_AHA_wheel()

map_T2._update_mask(map_T1)
map_DWI._update_mask(map_T1)
segment_16=map_T2.segment_16
maskFinal=np.zeros((Nx,Ny),dtype=int)
for nn,masktmp in enumerate(segment_16):
    maskFinal+=segment_16[nn] *(nn+1)
plt.imshow(maskFinal,vmax=18)
plt.show()

#%%




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
