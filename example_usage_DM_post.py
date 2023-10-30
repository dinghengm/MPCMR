# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libMapping_v12 import * # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 300

import pydicom 
#%%
CIRC_ID='CIRC_00382'
mapPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737')
mapList=[]
for dirpath,dirs,files in  os.walk(mapPath):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            mapList.append(path)
print(mapList)
#%%

num_slice=3
with open(mapList[0], 'rb') as inp:
    map_MP01 = pickle.load(inp)
print(f'{map_MP01.ID}')
with open(mapList[1], 'rb') as inp:
    map_MP02 = pickle.load(inp)
print(f'{map_MP02.ID}')
with open(mapList[2], 'rb') as inp:
    map_MP03 = pickle.load(inp)
print(f'{map_MP03.ID}')
cropzone=map_MP01.cropzone
# %%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_T1')
#CIRC_ID='CIRC_00302'

ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01 = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327)
MP01.cropzone=cropzone
MP01.go_crop()
MP01.go_resize(scale=2)
#fig,axs=MP01.imshow_corrected(ID='MP01_T1_raw',plot=False)

data_8000,_,_=readFolder(dicomPath=r'C:\Research\MRI\MP_EPI\CIRC_00382_22737_CIRC_00382_22737\MP02_T2\MR ep2d_MP01_TE_40_bright')
# %%
#Read one Slice
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=True,default=327,sigma=100,sortSlice=False)
data0=np.transpose(np.array(data['Slice0']),(1,2,0))
data0=np.expand_dims(data0,2)

MP01_0 = mapping(data=data0,CIRC_ID=CIRC_ID,ID='Slice0',valueList=valueDict['Slice0'],datasets=dcmDict['Slice0'])
MP01_0.path=dicomPath
MP01_0.cropzone=cropzone
MP01_0.go_crop()
MP01_0.valueList=valueDict['Slice0']
MP01_0.go_resize(scale=2)
MP01_0.imshow_corrected(ID=f'Slice0_raw',plot=True)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_0._delete(d=[4,6,-1,-2,0,2,9,10,12])
#%%
MP01_0_temp=np.copy(MP01_0._data)
MP01_0_regressed=decompose_LRT(MP01_0_temp)
list=MP01_0.valueList
MP01_0.imshow_corrected(volume=MP01_0_regressed,ID=f'Slice0_Regressed',valueList=list,plot=True)

Nx,Ny,Nz,_=np.shape(MP01_0_regressed)
MP01_0_temp_corrected_temp=np.copy(MP01_0_regressed)
for z in range(Nz):
    MP01_0_temp_corrected_temp[:,:,z,:]=MP01_0._coregister_elastix(MP01_0_regressed[:,:,z,:],MP01_0_temp[:,:,z,:])
MP01_0._data=MP01_0_temp_corrected_temp
MP01_0.imshow_corrected(ID=f'Slice0_Truncated_1',plot=True)

#%%
data1=np.transpose(np.array(data['Slice1']),(1,2,0))
data1=np.expand_dims(data1,2)
MP01_1 = mapping(data=data1,CIRC_ID=CIRC_ID,ID='Slice1',valueList=valueDict['Slice1'],datasets=dcmDict['Slice1'])
MP01_1.path=dicomPath
MP01_1.cropzone=cropzone
MP01_1.go_crop()
MP01_1.valueList=valueDict['Slice1']
MP01_1.go_resize(scale=2)
MP01_1.imshow_corrected(ID=f'Slice1_raw',plot=True)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_1._delete(d=[0,2,4,6,7,8,10,12,14,-2,-3])
#%%
MP01_1_temp=np.copy(MP01_1._data)
MP01_1_regressed=decompose_LRT(MP01_1_temp)
list=MP01_1.valueList
MP01_1.imshow_corrected(volume=MP01_1_regressed,ID=f'Slice1_Regressed',valueList=list,plot=True)

Nx,Ny,Nz,_=np.shape(MP01_1_regressed)
MP01_1_temp_corrected_temp=np.copy(MP01_1_regressed)
for z in range(Nz):
    MP01_1_temp_corrected_temp[:,:,z,:]=MP01_1._coregister_elastix(MP01_1_regressed[:,:,z,:],MP01_1_temp[:,:,z,:])
MP01_1._data=MP01_1_temp_corrected_temp
MP01_1.imshow_corrected(ID=f'Slice1_Truncated_1',plot=True)
#%%
data2=np.transpose(np.array(data['Slice2']),(1,2,0))
data2=np.expand_dims(data2,2)
MP01_2 = mapping(data=data2,CIRC_ID=CIRC_ID,ID='Slice2',valueList=valueDict['Slice2'],datasets=dcmDict['Slice2'])
MP01_2.path=dicomPath
MP01_2.cropzone=cropzone
MP01_2.go_crop()
MP01_2.valueList=valueDict['Slice2']
MP01_2.go_resize(scale=2)
MP01_2.imshow_corrected(ID=f'Slice2_raw',plot=True)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_2._delete(d=[8,-1,-2,0,2,4,6,10,12,14])
#%%
MP01_2_temp=np.copy(MP01_2._data)
MP01_2_regressed=decompose_LRT(MP01_2_temp)
list=MP01_2.valueList
MP01_2.imshow_corrected(volume=MP01_2_regressed,ID=f'Slice2_Truncated_1_Regressed',valueList=list,plot=True)

Nx,Ny,Nz,_=np.shape(MP01_2_regressed)
MP01_2_temp_corrected_temp=np.copy(MP01_2_regressed)
for z in range(Nz):
    MP01_2_temp_corrected_temp[:,:,z,:]=MP01_2._coregister_elastix(MP01_2_regressed[:,:,z,:],MP01_2_temp[:,:,z,:])
MP01_2._data=MP01_2_temp_corrected_temp
MP01_2.imshow_corrected(ID=f'Slice2_Truncated_1',plot=True)
# %%
#PLOT MOCO
for obj in [MP01_0,MP01_1,MP01_2]:
    Nz=obj.Nz
    A2=np.squeeze(np.copy(obj._data))
    A3 = A2/np.max(A2)*255
    img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{obj.ID}_moco_.gif')
    obj.createGIF(img_dir,A3,fps=5)
    obj._save_nib()
    list=[i+40 for i in obj.valueList]
    print(f'{obj.ID}:{list}')
#%%
#Save the map and show
crange=[0,3000]
cmap='magma'
path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'Slice0_nonduplicate_real.mat'))

map_data=map_data['T1']
MP01_0._map= np.expand_dims(map_data,axis=-1)
MP01_0.crange=crange
MP01_0.cmap=cmap
map_data=sio.loadmat(os.path.join(path,'Slice1_nonduplicate_real.mat'))
map_data=map_data['T1']
MP01_1._map= np.expand_dims(map_data,axis=-1)
map_data=sio.loadmat(os.path.join(path,'Slice2_nonduplicate_real.mat'))
map_data=map_data['T1']
MP01_2._map= np.expand_dims(map_data,axis=-1)
#%%
#Imshow
num_slice=3
figsize = (3.4*num_slice, 3)
plot=True
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
for sl,obj in enumerate([MP01_0,MP01_1,MP01_2]):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(np.flip(obj._map,axis=0), vmin=crange[0],vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_nonduplicate_real')
if plot:
    plt.savefig(img_dir)
plt.show()
#%%
#DRAW ROI
MP01_0.go_segment_LV(image_type="map",crange=crange)
MP01_0.save()
# look at stats
MP01_0.show_calc_stats_LV()
MP01_1.go_segment_LV(image_type="map",crange=crange)
MP01_1.save()
# look at stats
MP01_1.show_calc_stats_LV()
MP01_2.go_segment_LV(image_type="map",crange=crange)
MP01_2.save()
# look at stats
MP01_2.show_calc_stats_LV()

fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True,squeeze=True) 
for sl,obj in enumerate([MP01_0,MP01_1,MP01_2]):
    alpha = obj.mask_lv[..., 0] * 1.0
    base_im = obj._data[:, :, 0, 0]
    brightness = 0.8
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
    im = axes[sl].imshow(obj._map[...], alpha=alpha, vmin=crange[0],vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes[-1], shrink=0.95, pad=0.04, aspect=11)
mg_dir= os.path.join(os.path.dirname(dirpath),f'{obj.CIRC_ID}_nonduplicate_real_overlay')

plt.savefig(img_dir)
plt.show()  
#%%
# TEMP MASK
mask0=MP01_0.mask_lv
mask1=MP01_1.mask_lv
mask2=MP01_2.mask_lv
#%%
#Save the map and show
crange=[0,3000]
cmap='magma'
path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'Slice0_nonduplicate_make.mat'))

map_data=map_data['T1']
MP01_0._map= np.expand_dims(map_data,axis=-1)
MP01_0.crange=crange
MP01_0.cmap=cmap
map_data=sio.loadmat(os.path.join(path,'Slice1_nonduplicate_make.mat'))
map_data=map_data['T1']
MP01_1._map= np.expand_dims(map_data,axis=-1)
map_data=sio.loadmat(os.path.join(path,'Slice2_nonduplicate_make.mat'))
map_data=map_data['T1']
MP01_2._map= np.expand_dims(map_data,axis=-1)
#%%
#Imshow
num_slice=3
figsize = (3.4*num_slice, 3)
plot=True
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
for sl,obj in enumerate([MP01_0,MP01_1,MP01_2]):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(np.flip(obj._map,axis=0), vmin=crange[0],vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_nonduplicate_make')
if plot:
    plt.savefig(img_dir)
plt.show()
#%%
#DRAW ROI
MP01_0.mask_lv=mask0
MP01_1.mask_lv=mask0
MP01_2.mask_lv=mask0
MP01_0.show_calc_stats_LV()
MP01_1.show_calc_stats_LV()
MP01_2.show_calc_stats_LV()

fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True,squeeze=True) 
for sl,obj in enumerate([MP01_0,MP01_1,MP01_2]):
    alpha = obj.mask_lv[..., 0] * 1.0
    base_im = obj._data[:, :, 0, 0]
    brightness = 0.8
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
    im = axes[sl].imshow(obj._map[...], alpha=alpha, vmin=crange[0],vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes[-1], shrink=0.95, pad=0.04, aspect=11)
mg_dir= os.path.join(os.path.dirname(dirpath),f'{obj.CIRC_ID}_nonduplicate_make_overlay')

plt.savefig(img_dir)
plt.show()  
        
# %%
###########################################################################
##########################################################################
####################TEMP_Make_The_Last_TI_TIME###############################
data_8000,_,_=readFolder(dicomPath=r'C:\Research\MRI\MP_EPI\CIRC_00382_22737_CIRC_00382_22737\MP02_T2\MR ep2d_MP01_TE_40_bright')
# %%
#Read one Slice
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=True,default=327,sigma=100,sortSlice=False)
data0=np.transpose(np.array(data['Slice0']),(1,2,0))
data0=np.concatenate((data0,data_8000[:,:,0]),axis=-1)
data0=np.expand_dims(data0,2)
list=valueDict['Slice0'].append(8000)
MP01_0 = mapping(data=data0,CIRC_ID=CIRC_ID,ID='Slice0',valueList=list,datasets=dcmDict['Slice0'])
MP01_0.path=dicomPath
MP01_0.cropzone=cropzone
MP01_0.go_crop()
MP01_0.valueList=valueDict['Slice0']
MP01_0.go_resize(scale=2)
MP01_0.imshow_corrected(ID=f'Slice0_raw',plot=False)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_0._delete(d=[4,6,-2,-3,0,2,9,10,12])
#%%
MP01_0_temp=np.copy(MP01_0._data)
MP01_0_regressed=decompose_LRT(MP01_0_temp)
list=MP01_0.valueList
MP01_0.imshow_corrected(volume=MP01_0_regressed,ID=f'Slice0_Regressed',valueList=list,plot=False)

Nx,Ny,Nz,_=np.shape(MP01_0_regressed)
MP01_0_temp_corrected_temp=np.copy(MP01_0_regressed)
for z in range(Nz):
    MP01_0_temp_corrected_temp[:,:,z,:]=MP01_0._coregister_elastix(MP01_0_regressed[:,:,z,:],MP01_0_temp[:,:,z,:])
MP01_0._data=MP01_0_temp_corrected_temp
MP01_0.imshow_corrected(ID=f'Slice0_Truncated_1',plot=False)

#%%
data1=np.transpose(np.array(data['Slice1']),(1,2,0))
data1=np.concatenate((data1,data_8000[:,:,1]),axis=-1)

data1=np.expand_dims(data1,2)
list=valueDict['Slice1'].append(8000)
MP01_1 = mapping(data=data1,CIRC_ID=CIRC_ID,ID='Slice1',valueList=list,datasets=dcmDict['Slice1'])
MP01_1.path=dicomPath
MP01_1.cropzone=cropzone
MP01_1.go_crop()
MP01_1.valueList=valueDict['Slice1']
MP01_1.go_resize(scale=2)
MP01_1.imshow_corrected(ID=f'Slice1_raw',plot=False)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_1._delete(d=[0,2,4,6,7,8,10,12,14,-3,-4])
#%%
MP01_1_temp=np.copy(MP01_1._data)
MP01_1_regressed=decompose_LRT(MP01_1_temp)
list=MP01_1.valueList
MP01_1.imshow_corrected(volume=MP01_1_regressed,ID=f'Slice1_Regressed',valueList=list,plot=False)

Nx,Ny,Nz,_=np.shape(MP01_1_regressed)
MP01_1_temp_corrected_temp=np.copy(MP01_1_regressed)
for z in range(Nz):
    MP01_1_temp_corrected_temp[:,:,z,:]=MP01_1._coregister_elastix(MP01_1_regressed[:,:,z,:],MP01_1_temp[:,:,z,:])
MP01_1._data=MP01_1_temp_corrected_temp
MP01_1.imshow_corrected(ID=f'Slice1_Truncated_1',plot=False)
#%%
data2=np.transpose(np.array(data['Slice2']),(1,2,0))
data2=np.concatenate((data2,data_8000[:,:,2]),axis=-1)
list=valueDict['Slice2'].append(8000)
data2=np.expand_dims(data2,2)
MP01_2 = mapping(data=data2,CIRC_ID=CIRC_ID,ID='Slice2',valueList=list,datasets=dcmDict['Slice2'])
MP01_2.path=dicomPath
MP01_2.cropzone=cropzone
MP01_2.go_crop()
MP01_2.valueList=valueDict['Slice2']
MP01_2.go_resize(scale=2)
MP01_2.imshow_corrected(ID=f'Slice2_raw',plot=False)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_2._delete(d=[8,-2,-3,0,2,4,6,10,12,14])
#%%
MP01_2_temp=np.copy(MP01_2._data)
MP01_2_regressed=decompose_LRT(MP01_2_temp)
list=MP01_2.valueList
MP01_2.imshow_corrected(volume=MP01_2_regressed,ID=f'Slice2_Truncated_1_Regressed',valueList=list,plot=False)

Nx,Ny,Nz,_=np.shape(MP01_2_regressed)
MP01_2_temp_corrected_temp=np.copy(MP01_2_regressed)
for z in range(Nz):
    MP01_2_temp_corrected_temp[:,:,z,:]=MP01_2._coregister_elastix(MP01_2_regressed[:,:,z,:],MP01_2_temp[:,:,z,:])
MP01_2._data=MP01_2_temp_corrected_temp
MP01_2.imshow_corrected(ID=f'Slice2_Truncated_1',plot=False)
# %%
#PLOT MOCO
for ind,obj in enumerate([MP01_0,MP01_1,MP01_2]):
    Nz=obj.Nz
    A2=np.squeeze(np.copy(obj._data))
    A3 = A2/np.max(A2)*255
    img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{obj.ID}_make_moco_.gif')
    obj.createGIF(img_dir,A3,fps=5)
    obj._save_nib(ID=f'{obj.ID}_make')
    list=[i+40 for i in obj.valueList]
    print(f'{obj.ID}:{list}')
#%%
#Save the map and show
crange=[0,3000]
cmap='magma'
path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'Slice0.mat'))

map_data=map_data['T1']
MP01_0._map= np.expand_dims(map_data,axis=-1)
MP01_0.crange=crange
MP01_0.cmap=cmap
map_data=sio.loadmat(os.path.join(path,'Slice1.mat'))
map_data=map_data['T1']
MP01_1._map= map_data
map_data=sio.loadmat(os.path.join(path,'Slice2.mat'))
map_data=map_data['T1']
MP01_2._map= map_data
#%%
#Imshow
num_slice=3
figsize = (3.4*num_slice, 3)
plot=True
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
for sl,obj in enumerate([MP01_0,MP01_1,MP01_2]):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(np.flip(obj._map,axis=0), vmin=crange[0],vmax=crange[1], cmap=cmap)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{ID}')
if plot:
    plt.savefig(img_dir)
plt.show()

# %%
###########################################################################
##########################################################################
####################THE_Large Delay###############################
# %%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_T1')
#CIRC_ID='CIRC_00302'

ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01 = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327)
MP01.cropzone=cropzone
MP01.go_crop()
MP01.go_resize(scale=2)
#fig,axs=MP01.imshow_corrected(ID='MP01_T1_raw',plot=False)

data_8000,_,_=readFolder(dicomPath=r'C:\Research\MRI\MP_EPI\CIRC_00382_22737_CIRC_00382_22737\MP02_T2\MR ep2d_MP01_TE_40_bright')
# %%
#Read one Slice
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=True,default=725,sigma=250,sortSlice=False)
data0=np.transpose(np.array(data['Slice0']),(1,2,0))
data0=np.expand_dims(data0,2)
ID='Slice0_Delay_500'
MP01_0 = mapping(data=data0,CIRC_ID=CIRC_ID,ID=ID,valueList=valueDict['Slice0'],datasets=dcmDict['Slice0'])
MP01_0.path=dicomPath
MP01_0.cropzone=cropzone
MP01_0.go_crop()
MP01_0.valueList=valueDict['Slice0']
MP01_0.go_resize(scale=2)
MP01_0.imshow_corrected(ID=f'{ID}_raw',plot=True)
plt.rcParams.update({'axes.titlesize': 'small'})
# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_0._delete(d=[3,4,8,12])
#%%
MP01_0_temp=np.copy(MP01_0._data)
MP01_0_regressed=decompose_LRT(MP01_0_temp)
list=MP01_0.valueList
MP01_0.imshow_corrected(volume=MP01_0_regressed,ID=f'{ID}_Regressed',valueList=list,plot=True)

Nx,Ny,Nz,_=np.shape(MP01_0_regressed)
MP01_0_temp_corrected_temp=np.copy(MP01_0_regressed)
for z in range(Nz):
    MP01_0_temp_corrected_temp[:,:,z,:]=MP01_0._coregister_elastix(MP01_0_regressed[:,:,z,:],MP01_0_temp[:,:,z,:])
MP01_0._data=MP01_0_temp_corrected_temp
MP01_0.imshow_corrected(ID=f'{ID}_Truncated_1',plot=True)

#%%
data1=np.transpose(np.array(data['Slice1']),(1,2,0))
data1=np.expand_dims(data1,2)
ID='Slice1_Delay_500'
MP01_1 = mapping(data=data1,CIRC_ID=CIRC_ID,ID=ID,valueList=valueDict['Slice1'],datasets=dcmDict['Slice1'])
MP01_1.path=dicomPath
MP01_1.cropzone=cropzone
MP01_1.go_crop()
MP01_1.valueList=valueDict['Slice1']
MP01_1.go_resize(scale=2)
MP01_1.imshow_corrected(ID=f'{ID}_raw',plot=True)
plt.rcParams.update({'axes.titlesize': 'small'})
# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_1._delete(d=[3,4,8,11])
#%%
MP01_1_temp=np.copy(MP01_1._data)
MP01_1_regressed=decompose_LRT(MP01_1_temp)
list=MP01_1.valueList
MP01_1.imshow_corrected(volume=MP01_1_regressed,ID=f'{ID}_Regressed',valueList=list,plot=True)

Nx,Ny,Nz,_=np.shape(MP01_1_regressed)
MP01_1_temp_corrected_temp=np.copy(MP01_1_regressed)
for z in range(Nz):
    MP01_1_temp_corrected_temp[:,:,z,:]=MP01_1._coregister_elastix(MP01_1_regressed[:,:,z,:],MP01_1_temp[:,:,z,:])
MP01_1._data=MP01_1_temp_corrected_temp
MP01_1.imshow_corrected(ID=f'{ID}_Truncated_1',plot=True)
#%%
data2=np.transpose(np.array(data['Slice2']),(1,2,0))
data2=np.expand_dims(data2,2)
ID='Slice2_Delay_500'
MP01_2 = mapping(data=data2,CIRC_ID=CIRC_ID,ID=ID,valueList=valueDict['Slice2'],datasets=dcmDict['Slice2'])
MP01_2.path=dicomPath
MP01_2.cropzone=cropzone
MP01_2.go_crop()
MP01_2.valueList=valueDict['Slice2']
MP01_2.go_resize(scale=2)
MP01_2.imshow_corrected(ID=f'{ID}_raw',plot=True)
plt.rcParams.update({'axes.titlesize': 'small'})
# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_2._delete(d=[-1,-4])
#%%
MP01_2_temp=np.copy(MP01_2._data)
MP01_2_regressed=decompose_LRT(MP01_2_temp)
list=MP01_2.valueList
MP01_2.imshow_corrected(volume=MP01_2_regressed,ID=f'{ID}_Regressed',valueList=list,plot=True)

Nx,Ny,Nz,_=np.shape(MP01_2_regressed)
MP01_2_temp_corrected_temp=np.copy(MP01_2_regressed)
for z in range(Nz):
    MP01_2_temp_corrected_temp[:,:,z,:]=MP01_2._coregister_elastix(MP01_2_regressed[:,:,z,:],MP01_2_temp[:,:,z,:])
MP01_2._data=MP01_2_temp_corrected_temp
MP01_2.imshow_corrected(ID=f'{ID}_Truncated_1',plot=True)
# %%
#PLOT MOCO
for obj in [MP01_0,MP01_1,MP01_2]:
    Nz=obj.Nz
    A2=np.squeeze(np.copy(obj._data))
    A3 = A2/np.max(A2)*255
    img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{obj.ID}_moco_.gif')
    obj.createGIF(img_dir,A3,fps=5)
    obj._save_nib()
    list=[i+40 for i in obj.valueList]
    print(f'{obj.ID}:{list}')
# %%
