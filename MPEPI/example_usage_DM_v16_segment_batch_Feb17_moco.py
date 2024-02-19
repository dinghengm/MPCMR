###############Batch contour for paper: Nd~=36 * Nz= 3 * raw/moco=2 * subjects=13 = 2800
##############From  saved_ims_v2_Feb_5_2024/NULL
##############TO  saved_ims_v2_Feb_5_2024/NULL/xxxp.pickle
##############The data inside are masklv, mask_endo, mask_epi
#############Shape Nx,Ny,Nd
################
############## The Contour can save for later in DL


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
from roipoly import RoiPoly, MultiRoi
from skimage.transform import resize as imresize
from imgbasics import imcrop #for croping

matplotlib.rcParams['savefig.dpi'] = 400
plot=True
#%%
CIRC_ID_List=[446,429,419,405,398,382,381,373,457,472,486,498,500]
#CIRC_NUMBER=CIRC_ID_List[9]
CIRC_NUMBER=CIRC_ID_List[1]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','NULL',f'{CIRC_ID}')
img_save_dir=img_root_dir
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)
MP01_0=mapping(mapList[0])
MP01_1=mapping(mapList[1])
MP01_2=mapping(mapList[2])
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[5])
print(mapList)
dataDict={}
#%%
#Test Slice0

%matplotlib qt
MP01_List=[MP01_0,MP01_1,MP01_2]
slice_List=[]
ss=0
map01=MP01_List[ss]
data0MP02=MP02._data[:,:,ss,:]
data0MP03=MP03._data[:,:,ss,:]
data0MP02_raw=MP02._raw_data[:,:,ss,:]
data0MP03_raw=MP03._raw_data[:,:,ss,:]

dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
#dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
#dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
cmap='gray'
brightness=0.6
#%%
def draw_ROI_all(data,brightness=0.6,cmap='gray'):
        
    Nx,Ny,Nd=np.shape(data.squeeze())

    mask_endo = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
    mask_epi = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
    mask_lv = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
    for d in range(Nd):
        image = data[...,d].squeeze()
        roi_names=['endo','epi']
        fig = plt.figure()
        plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)
        plt.title('Nd: '+ str(d)+f'/{Nd}')
        fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
        multirois = MultiRoi(fig=fig, roi_names=roi_names)
        try:
            mask_endo[:,:,d]=multirois.rois['endo'].get_mask(image)
        except:
            continue

        mask_epi[:,:,d]=multirois.rois['epi'].get_mask(image)
        mask_lv[:,:,d]=mask_endo[:,:,d]^mask_epi[:,:,d]
       
    plt.close()
    return mask_endo,mask_epi,mask_lv
#%%
mask_endo,mask_epi,mask_lv=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap)
#%%
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data'),dataSlice0.squeeze())
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo'),mask_endo)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi'),mask_epi)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv'),mask_lv)
dataDict[f'Slice{ss}_data']=dataSlice0.squeeze()
dataDict[f'Slice{ss}_endo']=mask_endo
dataDict[f'Slice{ss}_epi']=mask_epi
dataDict[f'Slice{ss}_lv']=mask_lv

#%%
#Test on Raw
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)

new_data= MP02._crop_Auto(data=dataSlice0_raw)

mask_endo_raw,mask_epi,mask_lv=draw_ROI_all(dataSlice0_raw,brightness=brightness,cmap=cmap)


dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))
mask_endo_raw = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
mask_epi_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
mask_lv_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
%matplotlib qt


for d in range(Nd):
    image = dataSlice0_raw_new[...,d].squeeze()
    roi_names=['endo','epi']
    fig = plt.figure()
    plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)
    plt.title('Nd: '+ str(d)+f'/{Nd}')
    fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
    multirois = MultiRoi(fig=fig, roi_names=roi_names)
    mask_endo_raw[:,:,d]=multirois.rois['endo'].get_mask(image)
    mask_epi_raw[:,:,d]=multirois.rois['epi'].get_mask(image)
    mask_lv_raw[:,:,d]=mask_epi_raw[:,:,d]^mask_endo_raw[:,:,d]
    plt.close()
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data_raw'),dataSlice0_raw_new)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo_raw'),mask_endo_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi_raw'),mask_epi_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv_raw'),mask_lv_raw)
dataDict[f'Slice{ss}_data_raw']=dataSlice0_raw_new
dataDict[f'Slice{ss}_endo_raw']=mask_endo_raw
dataDict[f'Slice{ss}_epi_raw']=mask_epi_raw
dataDict[f'Slice{ss}_lv_raw']=mask_lv_raw
#%%
#############################################Slice1
%matplotlib qt
MP01_List=[MP01_0,MP01_1,MP01_2]
slice_List=[]
ss=1
map01=MP01_List[ss]
data0MP02=MP02._data[:,:,ss,:]
data0MP03=MP03._data[:,:,ss,:]
data0MP02_raw=MP02._raw_data[:,:,ss,:]
data0MP03_raw=MP03._raw_data[:,:,ss,:]

dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
#dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
#dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
cmap='gray'
brightness=0.8
Nx,Ny,Nd=np.shape(dataSlice0.squeeze())

mask_endo = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
mask_epi = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
mask_lv = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
%matplotlib qt
for d in range(Nd):
    image = dataSlice0[...,d].squeeze()
    roi_names=['endo','epi']
    fig = plt.figure()
    plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)
    plt.title('Nd: '+ str(d)+f'/{Nd}')
    fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
    multirois = MultiRoi(fig=fig, roi_names=roi_names)
    mask_endo[:,:,d]=multirois.rois['endo'].get_mask(image)
    mask_epi[:,:,d]=multirois.rois['epi'].get_mask(image)
    mask_lv[:,:,d]=mask_endo[:,:,d]^mask_epi[:,:,d]
    plt.close()
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data'),dataSlice0.squeeze())
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo'),mask_endo)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi'),mask_epi)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv'),mask_lv)
dataDict[f'Slice{ss}_data']=dataSlice0.squeeze()
dataDict[f'Slice{ss}_endo']=mask_endo
dataDict[f'Slice{ss}_epi']=mask_epi
dataDict[f'Slice{ss}_lv']=mask_lv

#%%
#Test on Raw Slice1
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)

new_data= MP02._crop_Auto(data=dataSlice0_raw)
dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))
mask_endo_raw = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
mask_epi_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
mask_lv_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
%matplotlib qt
for d in range(Nd):
    image = dataSlice0_raw_new[...,d].squeeze()
    roi_names=['endo','epi']
    fig = plt.figure()
    plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)    
    plt.title('Nd: '+ str(d)+f'/{Nd}')
    fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
    multirois = MultiRoi(fig=fig, roi_names=roi_names)
    mask_endo_raw[:,:,d]=multirois.rois['endo'].get_mask(image)
    mask_epi_raw[:,:,d]=multirois.rois['epi'].get_mask(image)
    mask_lv_raw[:,:,d]=mask_epi_raw[:,:,d]^mask_endo_raw[:,:,d]
    plt.close()
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data_raw'),dataSlice0_raw_new)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo_raw'),mask_endo_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi_raw'),mask_epi_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv_raw'),mask_lv_raw)

dataDict[f'Slice{ss}_data_raw']=dataSlice0_raw_new
dataDict[f'Slice{ss}_endo_raw']=mask_endo_raw
dataDict[f'Slice{ss}_epi_raw']=mask_epi_raw
dataDict[f'Slice{ss}_lv_raw']=mask_lv_raw

#%%
#############################################Slice2
%matplotlib qt
MP01_List=[MP01_0,MP01_1,MP01_2]
slice_List=[]
ss=2
brightness=0.8
map01=MP01_List[ss]
data0MP02=MP02._data[:,:,ss,:]
data0MP03=MP03._data[:,:,ss,:]
data0MP02_raw=MP02._raw_data[:,:,ss,:]
data0MP03_raw=MP03._raw_data[:,:,ss,:]

dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
#dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
#dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
cmap='gray'

Nx,Ny,Nd=np.shape(dataSlice0.squeeze())

mask_endo = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
mask_epi = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
mask_lv = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
%matplotlib qt
for d in range(Nd):
    image = dataSlice0[...,d].squeeze()
    roi_names=['endo','epi']
    fig = plt.figure()
    plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)
    plt.title('Nd: '+ str(d)+f'/{Nd}')
    fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
    multirois = MultiRoi(fig=fig, roi_names=roi_names)
    mask_endo[:,:,d]=multirois.rois['endo'].get_mask(image)
    mask_epi[:,:,d]=multirois.rois['epi'].get_mask(image)
    mask_lv[:,:,d]=mask_endo[:,:,d]^mask_epi[:,:,d]
    plt.close()
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data'),dataSlice0.squeeze())
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo'),mask_endo)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi'),mask_epi)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv'),mask_lv)
dataDict[f'Slice{ss}_data']=dataSlice0.squeeze()
dataDict[f'Slice{ss}_endo']=mask_endo
dataDict[f'Slice{ss}_epi']=mask_epi
dataDict[f'Slice{ss}_lv']=mask_lv

#%%
#Test on Raw Slice2
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)

new_data= MP02._crop_Auto(data=dataSlice0_raw)
dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))
mask_endo_raw = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
mask_epi_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
mask_lv_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
%matplotlib qt
for d in range(Nd):
    image = dataSlice0_raw_new[...,d].squeeze()
    roi_names=['endo','epi']
    fig = plt.figure()
    plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)    
    plt.title('Nd: '+ str(d)+f'/{Nd}')
    fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
    multirois = MultiRoi(fig=fig, roi_names=roi_names)
    mask_endo_raw[:,:,d]=multirois.rois['endo'].get_mask(image)
    mask_epi_raw[:,:,d]=multirois.rois['epi'].get_mask(image)
    mask_lv_raw[:,:,d]=mask_endo_raw[:,:,d]^mask_epi_raw[:,:,d]
    plt.close()
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data_raw'),dataSlice0_raw_new)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo_raw'),mask_endo_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi_raw'),mask_epi_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv_raw'),mask_lv_raw)

dataDict[f'Slice{ss}_data_raw']=dataSlice0_raw_new
dataDict[f'Slice{ss}_endo_raw']=mask_endo_raw
dataDict[f'Slice{ss}_epi_raw']=mask_epi_raw
dataDict[f'Slice{ss}_lv_raw']=mask_lv_raw
#%%
for key,val in dataDict.items():
    print(f'{key}:{np.shape(val)}')

#%%
#See the result
from skimage import measure
plt.figure()
for ss in range(3):
    mask_lv=dataDict[f'Slice{ss}_lv']
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
        plt.axis('off')
    plt.show()
#%%
import pickle
saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
with open(saveDict_name, 'wb') as fp:
    pickle.dump(dataDict, fp)
    print('dictionary saved successfully to file')



#%%
#Test on Raw Slice1
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)

new_data= MP02._crop_Auto(data=dataSlice0_raw)
dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))
mask_endo_raw = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
mask_epi_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
mask_lv_raw = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
%matplotlib qt
for d in range(Nd):
    image = dataSlice0_raw_new[...,d].squeeze()
    roi_names=['endo','epi']
    fig = plt.figure()
    plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)    
    plt.title('Nd: '+ str(d)+f'/{Nd}')
    fig.canvas.manager.set_window_title('Nd: '+ str(d)+f'/{Nd}')
    multirois = MultiRoi(fig=fig, roi_names=roi_names)
    mask_endo_raw[:,:,d]=multirois.rois['endo'].get_mask(image)
    mask_epi_raw[:,:,d]=multirois.rois['epi'].get_mask(image)
    mask_lv_raw[:,:,d]=mask_endo_raw[:,:,d]^mask_epi_raw[:,:,d]
    plt.close()
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data_raw'),dataSlice0_raw_new)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo_raw'),mask_endo_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi_raw'),mask_epi_raw)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv_raw'),mask_lv_raw)

dataDict[f'Slice{ss}_data']=dataSlice0_raw_new
dataDict[f'Slice{ss}_endo_raw']=mask_endo_raw
dataDict[f'Slice{ss}_epi_raw']=mask_epi_raw
dataDict[f'Slice{ss}_lv_raw']=mask_lv_raw
#%%
########################################################
#Test
dataDict1={}
MP01_List=[MP01_0,MP01_1,MP01_2]
for ss in range(3):
    map01=MP01_List[ss]
    bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
    Nx,Ny=np.shape(MP01_0._data[:,:,0,0])
    Nd-len(bval0)
    a=np.zeros((Nx,Ny,Nd))
    a=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data.npy'))
    b=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo.npy'))
    c=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi.npy'))
    d=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv.npy'))

    dataDict1[f'Slice{ss}_data']=a.squeeze()
    dataDict1[f'Slice{ss}_endo']=b
    dataDict1[f'Slice{ss}_epi']=c
    dataDict1[f'Slice{ss}_lv']=d



MP01_List=[MP01_0,MP01_1,MP01_2]
for ss in range(3):
    map01=MP01_List[ss]
    bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
    Nx,Ny=np.shape(MP01_0._data[:,:,0,0])
    Nd-len(bval0)
    a=np.zeros((Nx,Ny,Nd))
    a=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data_raw.npy'))
    b=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo_raw.npy'))
    c=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi_raw.npy'))
    d=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv_raw.npy'))

    dataDict1[f'Slice{ss}_data_raw']=a
    dataDict1[f'Slice{ss}_endo_raw']=b
    dataDict1[f'Slice{ss}_epi_raw']=c
    dataDict1[f'Slice{ss}_lv_raw']=d


#%%
for key,val in dataDict1.items():
    print(f'{key}:{np.shape(val)}')

#%%
#bmode
def bmode(obj,data=None,ID=None,x=None,y=None,plot=False,path=None):
    if path is None:
        path=os.path.dirname(obj.path)
    if ID is None:
        ID=obj.ID
    if data is None:
        data=obj.data
    
    Nx,Ny,Nz,Nd=np.shape(data)
    if x==None and y==None:
        x=int(np.shape(data)[0]/2)
    if x is not None:
        A2=np.zeros((Ny,Nz,Nd),dtype=np.float64)
        A2=data[x,:,:,:]
    elif y is not None:
        A2=np.zeros((Nx,Nz,Nd),dtype=np.float64)
        A2=data[:,y,:,:]
    if Nz !=1:
        fig,axs=plt.subplots(2,Nz)
        ax=axs.ravel()
        for i in range(Nz):
            ax[1,i].imshow(data[...,i,0],cmap='gray')
            if x is not None:
                ax[1,i].axhline(y=x, color='r', linestyle='-')
            if y is not None:
                ax[1,i].axvline(x=y, color='r', linestyle='-')
            ax[2,i].imshow(A2[...,i,:],cmap='gray')
            ax[2,i].set_title(f'z={i}')
            ax[i].axis('off')
    elif Nz==1:
        plt.subplot(121)
        plt.imshow(data.squeeze()[...,0],cmap='gray')
        if x is not None:
            plt.axhline(y=x, color='r', linestyle='-')
        if y is not None:
            plt.axvline(x=y, color='r', linestyle='-')
        plt.subplot(122)
        A3=np.squeeze(A2)
        plt.imshow(A3,cmap='gray')
        #plt.axis('off')
    if plot==True:
        dir=os.path.join(path,ID)
        plt.savefig(dir)
    plt.show()
for ss in range(3):
    print('raw')
    bmode(MP01_0,data=dataDict1[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],plot=True,ID=f'Slice{ss}_bmode_raw',path=img_save_dir)
    print('moco')
    bmode(MP01_0,data=dataDict1[f'Slice{ss}_data'][:,:,np.newaxis,:],plot=True,ID=f'Slice{ss}_bmode',path=img_save_dir)
#MP02.bmode(data=dataDict1[f'Slice{ss}_data'][:,:,np.newaxis,:])

#%%
from skimage import measure
%matplotlib inline
plt.figure()

for ss in range(3):
    plt.subplot(3,2,2*ss+1)
    Nd_01=np.shape(MP01_List[ss])[-1]
    image_data=dataDict1[f'Slice{ss}_data']
    mask_lv=dataDict1[f'Slice{ss}_lv']
    plt.imshow(image_data[...,Nd_01].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
        plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        plt.axis('off')

for ss in range(3):
    plt.subplot(3,2,2*ss+2)
    Nd_01=np.shape(MP01_List[ss])[-1]
    image_data=dataDict1[f'Slice{ss}_data_raw']
    mask_lv=dataDict1[f'Slice{ss}_lv_raw']
    plt.imshow(image_data[...,Nd_01].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
        plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        
        plt.axis('off')
plt.savefig(os.path.join(img_save_dir,f'contours'))
plt.show()

# %%
#EPI
for ss in range(3):
    plt.figure()
    mask_lv=dataDict1[f'Slice{ss}_lv_raw']
    mask_endo_raw=dataDict1[f'Slice{ss}_endo_raw']
    mask_epi_raw=dataDict1[f'Slice{ss}_epi_raw']
    Nd=np.shape(mask_epi_raw)[-1]
    for d in range(Nd):
        mask_lv[:,:,d]=mask_endo_raw[:,:,d]^mask_epi_raw[:,:,d]
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
        plt.axis('off')
    dataDict1[f'Slice{ss}_lv_raw']=mask_lv
    plt.show()

# %%
#Test load
saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
with open(saveDict_name, 'rb') as inp:
    dataDict2 = pickle.load(inp)
print('dictionary Load successfully to file')


# %%
