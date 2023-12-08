#%%
import numpy as np
import os
import pydicom 
from pydicom.filereader import read_dicomdir
#from CIRC_tools import imshowgrid, uigetfile, toc
from pathlib import Path
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tensorly.decomposition import tucker
import tensorly as tl
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
import time
import multiprocessing
import imageio # for gif
from roipoly import RoiPoly, MultiRoi
from matplotlib import pyplot as plt #for ROI poly and croping
from imgbasics import imcrop #for croping
from tqdm.auto import tqdm # progress bar
from ipyfilechooser import FileChooser # ui get file
import pickle # to save diffusion object
import fnmatch # this is for string comparison dicomread
import pandas as pd
from skimage.transform import resize as imresize
import nibabel as nib
import scipy.io as sio


try:
    from numba import njit #super fast C-like calculation
    _global_bNumba_support = True
except:
    print('does not have numba library ... slower calculations only')
    _global_bNumba_support = False


from libMapping import mapping,readFolder  

%matplotlib inline
#Read the file:

#%%
CIRC_ID='CIRC00303'
path=r'C:\Users\MAID2\Desktop\TEMP\CIRC00303'
#CIRC_ID='CIRC00292'
#path=r'C:\Users\MAID2\Desktop\TEMP\CIRC00292'
mapList=[]
for dirpath,dirs,files in  os.walk(path):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('.mapping'):
            mapList.append(path)
print(mapList)
#%%

num_slice=3
with open(mapList[0], 'rb') as inp:
    map_MP01 = pickle.load(inp)
print(f'{map_MP01.ID}')
with open(mapList[2], 'rb') as inp:
    map_MP02 = pickle.load(inp)
print(f'{map_MP02.ID}')
with open(mapList[5], 'rb') as inp:
    map_MP03 = pickle.load(inp)
print(f'{map_MP03.ID}')
with open(mapList[2], 'rb') as inp:
    map_T1 = pickle.load(inp)
print(f'{map_T1.ID}')
with open(mapList[1], 'rb') as inp:
    map_T1_FB = pickle.load(inp)
print(f'{map_T1_FB.ID}')

with open(mapList[4], 'rb') as inp:
    map_T2 = pickle.load(inp)
print(f'{map_T2.ID}')
with open(mapList[3], 'rb') as inp:
    map_T2_FB = pickle.load(inp)
print(f'{map_T2_FB.ID}')

root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
img_dir= os.path.join(root_dir,f'{CIRC_ID}_ALL')
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
#%%
CIRC_ID='CIRC00303'
path=r'C:\Users\MAID2\Desktop\TEMP\CIRC00303'
#CIRC_ID='CIRC00292'
#path=r'C:\Users\MAID2\Desktop\TEMP\CIRC00292'
mapList=[]
for dirpath,dirs,files in  os.walk(path):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('.mapping'):
            mapList.append(path)
print(mapList)
#%%

num_slice=3
with open(mapList[0], 'rb') as inp:
    map_MP01 = pickle.load(inp)
print(f'{map_MP01.ID}')
with open(r'C:\Users\MAID2\Desktop\TEMP\CIRC00303\MP02_2_np.mapping', 'rb') as inp:
    map_MP02 = pickle.load(inp)
print(f'{map_MP02.ID}')
with open(mapList[3], 'rb') as inp:
    map_MP03 = pickle.load(inp)
print(f'{map_MP03.ID}')
with open(mapList[5], 'rb') as inp:
    map_T1 = pickle.load(inp)
print(f'{map_T1.ID}')
with open(mapList[4], 'rb') as inp:
    map_T1_FB = pickle.load(inp)
print(f'{map_T1_FB.ID}')

with open(mapList[8], 'rb') as inp:
    map_T2 = pickle.load(inp)
print(f'{map_T2.ID}')
with open(mapList[6], 'rb') as inp:
    map_T2_FB = pickle.load(inp)
print(f'{map_T2_FB.ID}')

root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
img_dir= os.path.join(root_dir,f'{CIRC_ID}_ALL')
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

#%%
#Re SAVE
map_MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3000])
map_T1.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3000])
map_T1_FB.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3000])
map_MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,150])
map_T2.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,150])
map_T2_FB.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,150])
map_MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3])



#%%
from skimage.transform import resize as imresize
#%%
plt.close()
figsize = (3.4*num_slice, 3*num_slice)
newshape0,newshape1,*rest=np.shape(map_MP01._map)
for z in range(num_slice):
    fig,axs=plt.subplots(3,3,figsize=figsize)
    im1=axs[0,0].imshow(np.flip(map_MP01._map[:,:,z],axis=0),cmap='magma',vmin=0,vmax=3000)
    #im1=axs[0,0].imshow(map_MP01._map[:,:,z],cmap='magma',vmin=0,vmax=3000)
    axs[0,0].axis('off')
    axs[0,0].set_title('MP-EPI-T1')
    #im2=axs[0,1].imshow(map_MP02._map[:,:,z],cmap='viridis',vmin=0,vmax=150)
    im2=axs[0,1].imshow(np.flip(map_MP02._map[:,:,z],axis=0),cmap='viridis',vmin=0,vmax=150)
    axs[0,1].axis('off')
    axs[0,1].set_title('MP-EPI-T1')
    #im3=axs[0,2].imshow(map_MP03._map[:,:,z],cmap='hot',vmin=0,vmax=3)
    im3=axs[0,2].imshow(np.flip(map_MP03._map[:,:,z],axis=0),cmap='hot',vmin=0,vmax=3)
    axs[0,2].axis('off')
    axs[0,2].set_title('MP-EPI-ADC')
    axs[1,0].imshow(imresize(map_T1._map[:,:,z],(newshape0,newshape1)),cmap='magma',vmin=0,vmax=3000)
    #axs[1,0].imshow(map_MP01._resize(map_T1._map[:,:,z],newshape=np.shape(map_MP01._map[:,:,z])),cmap='magma',vmin=0,vmax=3000)
    axs[1,0].axis('off')
    axs[1,0].set_title('T1-MOLLI')
    axs[2,0].imshow(imresize(map_T1_FB._map[:,:,z],(newshape0,newshape1)),cmap='magma',vmin=0,vmax=3000)
    axs[2,0].axis('off')
    axs[2,0].set_title('T1-MOLLI-FB')
    axs[1,1].imshow(imresize(map_T2._map[:,:,z],(newshape0,newshape1)),cmap='viridis',vmin=0,vmax=150)
    axs[1,1].axis('off')
    axs[1,1].set_title('T2-FLASH')

    axs[2,1].imshow(imresize(map_T2_FB._map[:,:,z],(newshape0,newshape1)),cmap='viridis',vmin=0,vmax=150)
    axs[2,1].set_title('T2-FLASH-FB')
    axs[2,1].axis('off')
    axs[1,2].axis('off')
    axs[2,2].axis('off')

    #cb_ax=fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar1=fig.colorbar(im1,ax=axs[0,0], shrink=1, pad=0.018, aspect=11)
    cbar2=fig.colorbar(im2,ax=axs[0,1], shrink=1, pad=0.018, aspect=11)
    cbar3=fig.colorbar(im3,ax=axs[0,2], shrink=1, pad=0.018, aspect=11)
    if os.path.isfile(img_dir):
        plt.show()
        print('The file exists')
    else:
        plt.savefig(img_dir+'_'+str(z))
        plt.show()
        print('Save Fig Successfully')

#%%
%matplotlib qt
T1_mapping_All,valueList,DicomList=readFolder(r'C:\Research\MRI\MP_EPI\CIRC_00303_22737_CIRC_00303_22737\MR t1map_long_t1_3slice_8mm_150_gap_MOCO_T1',sortBy='seriesNumber')
T2_mapping_All,valueList,DicomList=readFolder(r'C:\Research\MRI\MP_EPI\CIRC_00303_22737_CIRC_00303_22737\MR t2map_flash_3slice_8mm_150_gap_MOCO_T2',sortBy='seriesNumber')

T2_All=mapping(T2_mapping_All)
T2_All.cropzone=[]
T2_All.go_crop()
T1_All=mapping(T1_mapping_All)
T1_All.cropzone=[]
T1_All.go_crop()
#%%

figsize = (8,8)
newshape0,newshape1,*rest=np.shape(map_MP01._map)
fig,axs=plt.subplots(3,3,figsize=figsize,constrained_layout=True)
for z in range(3):
    #im1=axs[0,Z].imshow(np.flip(map_MP01._map[:,:,z],axis=0),cmap='magma',vmin=0,vmax=3000)
    im1=axs[z,0].imshow(map_MP01._map[:,:,z],cmap='magma',vmin=0,vmax=3000)
    axs[0,z].axis('off')
    #axs[0,1].set_ylabel('MP-EPI-T1')
    im2=axs[z,1].imshow(map_MP02._map[:,:,z],cmap='viridis',vmin=0,vmax=150)
    #im2=axs[0,1].imshow(np.flip(map_MP02._map[:,:,z],axis=0),cmap='viridis',vmin=0,vmax=150)
    axs[1,z].axis('off')
    #axs[1,1].set_ylabel('MP-EPI-T2')
    im3=axs[z,2].imshow(map_MP03._map[:,:,z],cmap='hot',vmin=0,vmax=3)
    #im3=axs[0,2].imshow(np.flip(map_MP03._map[:,:,z],axis=0),cmap='hot',vmin=0,vmax=3)
    axs[2,z].axis('off')
    #axs[2,1].set_ylabel('MP-EPI-ADC')
    #axs[0,2].set_title('Apex')
    #axs[0,1].set_title('Mid')
    #axs[0,0].set_title('Base')
    #cb_ax=fig.add_axes([0.83, 0.1, 0.02, 0.8])

cbar1=fig.colorbar(im1,ax=axs[2,0], shrink=0.6, pad=0.018, aspect=11,orientation = "horizontal")
cbar2=fig.colorbar(im2,ax=axs[2,1], shrink=0.8, pad=0.018, aspect=11,orientation = "horizontal")
cbar3=fig.colorbar(im3,ax=axs[2,2], shrink=0.8, pad=0.018, aspect=11,orientation = "horizontal")
plt.show()
#%%
plt.close()
from mpl_toolkits.axes_grid1 import make_axes_locatable
%matplotlib inline
figsize = (4, 4)
newshape0,newshape1,*rest=np.shape(map_MP01._map)
fig,axs=plt.subplots(3,4,figsize=figsize)
for z in range(3):
    #im1=axs[0,Z].imshow(np.flip(map_MP01._map[:,:,z],axis=0),cmap='magma',vmin=0,vmax=3000)
    im1=axs[0,1+z].imshow(map_MP01._map[:,:,z],cmap='magma',vmin=0,vmax=3000)
    axs[0,1+z].axis('off')
    axs[0,1].set_ylabel('MP-EPI-T1')
    im2=axs[1,1+z].imshow(map_MP02._map[:,:,z],cmap='viridis',vmin=0,vmax=150)
    #im2=axs[0,1].imshow(np.flip(map_MP02._map[:,:,z],axis=0),cmap='viridis',vmin=0,vmax=150)
    axs[1,1+z].axis('off')
    axs[1,1].set_ylabel('MP-EPI-T2')
    im3=axs[2,1+z].imshow(map_MP03._map[:,:,z],cmap='hot',vmin=0,vmax=3)
    #im3=axs[0,2].imshow(np.flip(map_MP03._map[:,:,z],axis=0),cmap='hot',vmin=0,vmax=3)
    axs[2,1+z].axis('off')
    axs[2,1].set_ylabel('MP-EPI-ADC')
axs[0,1].set_title('Apex')
axs[0,2].set_title('Mid')
axs[0,3].set_title('Base')
    #cb_ax=fig.add_axes([0.83, 0.1, 0.02, 0.8])
im1=axs[0,0].imshow(T1_All._data[:,:,0],cmap='magma',vmin=0,vmax=3000)
#im2=axs[0,1].imshow(np.flip(map_MP02._map[:,:,z],axis=0),cmap='viridis',vmin=0,vmax=150)
axs[0,0].axis('off')
axs[0,0].set_title('T1-MOLLI')
im2=axs[1,0].imshow(T2_All._data[:,:,0]/10,cmap='viridis',vmin=0,vmax=150)
#im3=axs[0,2].imshow(np.flip(map_MP03._map[:,:,z],axis=0),cmap='hot',vmin=0,vmax=3)
axs[1,0].axis('off')
axs[1,0].set_title('T2-FLASH')
axs[2,0].axis('off')
#cbar1=fig.colorbar(im1,ax=axs[0,3], shrink=1, pad=0.018, aspect=11)
#cbar2=fig.colorbar(im2,ax=axs[1,3], shrink=1, pad=0.018, aspect=11)
#cbar3=fig.colorbar(im3,ax=axs[2,3], shrink=1, pad=0.018, aspect=11)
plt.show()
#%%
fig.colorbar(im1, orientation="horizontal", pad = 0.4)
plt.show()
fig.colorbar(im2, orientation="horizontal", pad = 0.4)
plt.show()
#%%
if os.path.isfile(img_dir):
    plt.show()
    print('The file exists')
else:
    plt.savefig(img_dir+'_'+str(z))
    plt.show()
    print('Save Fig Successfully')
# %%
#Show the data in one slice and in mosaic
A1=np.copy(map_MP01._data)
A2=np.copy(map_MP02._data)
A3=np.copy(map_MP03._data)

for i in range(num_slice):
    #A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:]) #original data
    A1[:,:,i,:] = A1[...,i,:]/np.max(A1[...,i,:])*255
    #A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:]) #Processed data
    A2[:,:,i,:] = A2[...,i,:]/np.max(A2[...,i,:])*255
    A3[:,:,i,:] = A3[...,i,:]/np.max(A3[...,i,:])*255
#A1_combine=np.concatenate((A1[:,:,slice_read,0:-1:2]),axis=-1)
#A2_combine=np.concatenate((A2[:,:,slice_read,0:6]),axis=-1)
#A3_combine=np.concatenate((A3[:,:,slice_read,0:6]),axis=-1)

# %%
plt.style.use('dark_background')
fig,axs=plt.subplots(3,6,figsize=(7,5))
#CIRC00302
slice_read=0
readlist=[0,2,3,4,6,9]
readlist=[0,2,3,4,5,9]
titime=[70,300,500,800,900,1200,1500,2180,2780,8000]


tetime=[30,40,50,60,80,100]
dwi=['b50x','b50y','b50z','b500x','b500y','b500z']

for d in range(6):
    axs[0,d].imshow(A1[:,:,slice_read,readlist[d]],cmap='gray')
    axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,d].axis('off')
    axs[1,d].imshow(A2[:,:,slice_read,d],cmap='gray')
    axs[1,d].set_title(f'TE={tetime[d]}ms',fontsize=5)
    axs[1,d].axis('off')
    axs[2,d].imshow(A3[:,:,slice_read,d],cmap='gray')
    axs[2,d].set_title(f'{dwi[d]}',fontsize=5)
    axs[2,d].axis('off')
# %%
CIRC_ID='CIRC00292'
path=r'C:\Users\MAID2\Desktop\TEMP\CIRC00292'
mapList=[]
for dirpath,dirs,files in  os.walk(path):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            mapList.append(path)
print(mapList)
num_slice=3
with open(mapList[0], 'rb') as inp:
    map_MP01 = pickle.load(inp)
print(f'{map_MP01.ID}')
with open(mapList[-2], 'rb') as inp:
    map_MP02 = pickle.load(inp)
print(f'{map_MP02.ID}')
with open(mapList[-1], 'rb') as inp:
    map_MP03 = pickle.load(inp)
print(f'{map_MP03.ID}')
with open(mapList[2], 'rb') as inp:
    map_T1 = pickle.load(inp)
print(f'{map_T1.ID}')
with open(mapList[1], 'rb') as inp:
    map_T1_FB = pickle.load(inp)
print(f'{map_T1_FB.ID}')

with open(mapList[4], 'rb') as inp:
    map_T2 = pickle.load(inp)
print(f'{map_T2.ID}')
with open(mapList[3], 'rb') as inp:
    map_T2_FB = pickle.load(inp)
print(f'{map_T2_FB.ID}')
#%%
root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
img_dir= os.path.join(root_dir,f'{CIRC_ID}_ALL')
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
A1=np.copy(map_MP01._data)
A2=np.copy(map_MP02._data)
A3=np.copy(map_MP03._data)
for i in range(num_slice):
    #A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:]) #original data
    A1[:,:,i,:] = A1[...,i,:]/np.max(A1[...,i,:])*255
    #A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:]) #Processed data
    A2[:,:,i,:] = A2[...,i,:]/np.max(A2[...,i,:])*255
    A3[:,:,i,:] = A3[...,i,:]/np.max(A3[...,i,:])*255

plt.style.use('dark_background')

fig,axs=plt.subplots(3,6,figsize=(7,5))
#CIRC00292
slice_read=1
readlist=[0,2,3,4,6,9]
readlist=[0,3,4,6,7,9]
titime=[70,200,300,500,800,900,1200,1500,2180,2780]


tetime=[30,40,50,60,80,100]
dwi=['b50x','b50y','b50z','b500x','b500y','b500z']

for d in range(6):
    axs[0,d].imshow(A1[:,:,slice_read,readlist[d]],cmap='gray')
    axs[0,d].set_title(f'TI={titime[readlist[d]]}',fontsize=5)
    axs[0,d].axis('off')
    axs[1,d].imshow(A2[:,:,slice_read,d],cmap='gray')
    axs[1,d].set_title(f'TE={tetime[d]}',fontsize=5)
    axs[1,d].axis('off')
    axs[2,d].imshow(A3[:,:,slice_read,d],cmap='gray')
    axs[2,d].set_title(f'{dwi[d]}',fontsize=5)
    axs[2,d].axis('off')
# %%
