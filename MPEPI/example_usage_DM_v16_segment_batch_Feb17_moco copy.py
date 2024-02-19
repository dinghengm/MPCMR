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
import pickle
from skimage import measure
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
#%%
####################405 has to regenerate MP03-DWI
CIRC_ID_List=[446,429,419,405,398,382,381,373,457,472,486,498,500]
#CIRC_NUMBER=CIRC_ID_List[9]
CIRC_NUMBER=CIRC_ID_List[4]
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
ss=0
map01=MP01_List[ss]
data0MP02=MP02._data[:,:,ss,:]
data0MP03=MP03._data[:,:,ss,:]
data0MP02_raw=MP02._raw_data[:,:,ss,:]
data0MP03_raw=MP03._raw_data[:,:,ss,:]

dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
#dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
#dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
cmap='gray'
brightness=0.6
Nx,Ny,Nd=np.shape(dataSlice0.squeeze())
#%%
def draw_ROI_all(data,brightness=0.6,cmap='gray',d=-1):
    
    Nx,Ny,Nd=np.shape(data.squeeze())
    if d==-1:
        slices=range(Nd)
    else:
        if type(d) == int:
            slices = [d]
        else:
            slices = np.copy(d)
    mask_endo = np.full((Nx,Ny,Nd), False, dtype=bool) # [None]*self.Nz
    mask_epi = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
    mask_lv = np.full((Nx,Ny,Nd), False, dtype=bool) #[None]*self.Nz
    try:
        for dd in slices:
            image = data[...,dd].squeeze()
            roi_names=['endo','epi']
            fig = plt.figure()
            plt.imshow(image, cmap=cmap,vmax=np.max(image)*brightness)
            plt.title('Nd: '+ str(dd)+f'/{Nd}')
            fig.canvas.manager.set_window_title('Nd: '+ str(dd)+f'/{Nd}')
            multirois = MultiRoi(fig=fig, roi_names=roi_names)
            try:
                mask_endo[:,:,dd]=multirois.rois['endo'].get_mask(image)
            except:
                continue

            mask_epi[:,:,dd]=multirois.rois['epi'].get_mask(image)
            mask_lv[:,:,dd]=mask_endo[:,:,dd]^mask_epi[:,:,dd]
        
        plt.close()
    except:
        return mask_endo,mask_epi,mask_lv
    return mask_endo,mask_epi,mask_lv

def plot_ROI(data,mask_lv):
    image_data=data
    plt.figure()
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    Nd=np.shape(image_data)[-1]
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)
        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,label=f'{d}',linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,label=f'{d}', linewidth=0.7)
        except:
            continue
    plt.legend()
    plt.axis('off')

#%%
%matplotlib qt
mask_endo,mask_epi,mask_lv=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap)
#%%
%matplotlib qt
d=[7]
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv[:,:,dd]=mask_lv_tmp[:,:,ind]

#%%
%matplotlib inline
plot_ROI(dataSlice0,mask_lv)

#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data'),dataSlice0.squeeze())
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo'),mask_endo)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi'),mask_epi)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv'),mask_lv)
dataDict[f'Slice{ss}_data']=dataSlice0.squeeze()
dataDict[f'Slice{ss}_endo']=mask_endo
dataDict[f'Slice{ss}_epi']=mask_epi
dataDict[f'Slice{ss}_lv']=mask_lv
#Test on Raw
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
_,_,Nd=np.shape(dataSlice0_raw.squeeze())
new_data= MP02._crop_Auto(data=dataSlice0_raw)
%matplotlib qt
dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))


#%%

mask_endo_raw,mask_epi_raw,mask_lv_raw=draw_ROI_all(dataSlice0_raw_new,brightness=brightness,cmap=cmap)

#%%
d=[12,15]
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(dataSlice0_raw_new,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo_raw[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi_raw[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv_raw[:,:,dd]=mask_lv_tmp[:,:,ind]

#%%
%matplotlib inline
plot_ROI(dataSlice0_raw_new,mask_lv_raw)


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

ss=1
map01=MP01_List[ss]
data0MP02=MP02._data[:,:,ss,:]
data0MP03=MP03._data[:,:,ss,:]
data0MP02_raw=MP02._raw_data[:,:,ss,:]
data0MP03_raw=MP03._raw_data[:,:,ss,:]
dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
#dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
#dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
cmap='gray'
brightness=1.2
Nx,Ny,Nd=np.shape(dataSlice0.squeeze())

#%%
%matplotlib qt
mask_endo,mask_epi,mask_lv=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap)
#%%
%matplotlib qt
d=range(17,Nd)
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv[:,:,dd]=mask_lv_tmp[:,:,ind]

#%%
%matplotlib inline
plot_ROI(dataSlice0,mask_lv)


#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data'),dataSlice0.squeeze())
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo'),mask_endo)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi'),mask_epi)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv'),mask_lv)
dataDict[f'Slice{ss}_data']=dataSlice0.squeeze()
dataDict[f'Slice{ss}_endo']=mask_endo
dataDict[f'Slice{ss}_epi']=mask_epi
dataDict[f'Slice{ss}_lv']=mask_lv

#Test on Raw
dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
_,_,Nd=np.shape(dataSlice0_raw.squeeze())
new_data= MP02._crop_Auto(data=dataSlice0_raw)
%matplotlib qt
dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))

#%%
%matplotlib qt
mask_endo_raw,mask_epi_raw,mask_lv_raw=draw_ROI_all(dataSlice0_raw_new,brightness=brightness,cmap=cmap)

#%%
#####Redraw ROI
%matplotlib qt
d=range(17,Nd)
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(dataSlice0_raw_new,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo_raw[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi_raw[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv_raw[:,:,dd]=mask_lv_tmp[:,:,ind]

#%%
%matplotlib inline
plot_ROI(dataSlice0_raw_new,mask_lv_raw)




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

ss=2
brightness=1.2

map01=MP01_List[ss]
data0MP02=MP02._data[:,:,ss,:]
data0MP03=MP03._data[:,:,ss,:]
data0MP02_raw=MP02._raw_data[:,:,ss,:]
data0MP03_raw=MP03._raw_data[:,:,ss,:]

dataSlice0=np.concatenate((map01._data,data0MP02[:,:,np.newaxis,:],data0MP03[:,:,np.newaxis,:]),axis=-1)
#dataSlice1_raw=np.concatenate((MP01_1._raw_data,MP02._raw_data[:,:,1,:],MP03._raw_data[:,:,1,:]))
#dataSlice2_raw=np.concatenate((MP01_2._raw_data,MP02._raw_data[:,:,2,:],MP03._raw_data[:,:,2,:]))
bval0=list(map01.valueList+MP02.valueList)+list(MP03.bval)
cmap='gray'
Nx,Ny,Nd=np.shape(dataSlice0.squeeze())
#%%
%matplotlib qt
mask_endo,mask_epi,mask_lv=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap)
#%%
%matplotlib qt
d=[18,20]
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(dataSlice0,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv[:,:,dd]=mask_lv_tmp[:,:,ind]

#%%
%matplotlib inline
plot_ROI(dataSlice0,mask_lv)
#%%
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data'),dataSlice0.squeeze())
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo'),mask_endo)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi'),mask_epi)
np.save(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv'),mask_lv)
dataDict[f'Slice{ss}_data']=dataSlice0.squeeze()
dataDict[f'Slice{ss}_endo']=mask_endo
dataDict[f'Slice{ss}_epi']=mask_epi
dataDict[f'Slice{ss}_lv']=mask_lv


dataSlice0_raw=np.concatenate((map01._raw_data,data0MP02_raw[:,:,np.newaxis,:],data0MP03_raw[:,:,np.newaxis,:]),axis=-1)
_,_,Nd=np.shape(dataSlice0_raw.squeeze())
new_data= MP02._crop_Auto(data=dataSlice0_raw)
%matplotlib qt
dataSlice0_raw_new=np.zeros((Nx,Ny,Nd))
for d in range(Nd):
    dataSlice0_raw_new[...,d]=imresize(new_data.squeeze()[...,d],(Nx,Ny))


#%%
%matplotlib qt
mask_endo_raw,mask_epi_raw,mask_lv_raw=draw_ROI_all(dataSlice0_raw_new,brightness=brightness,cmap=cmap)

#%%
#####Redraw ROI
%matplotlib qt
d=range(17,Nd)
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(dataSlice0_raw_new,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo_raw[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi_raw[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv_raw[:,:,dd]=mask_lv_tmp[:,:,ind]

#%%
%matplotlib inline
plot_ROI(dataSlice0_raw_new,mask_lv_raw)

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
#%%

saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
with open(saveDict_name, 'wb') as fp:
    pickle.dump(dataDict, fp)
    print('dictionary saved successfully to file')


#%%
#bmode
def bmode(data=None,ID=None,x=None,y=None,plot=False,path=None):

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
    bmode(data=dataDict[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],
            plot=True,
            ID=f'Slice{ss}_bmode_raw',
            path=img_save_dir)
    print('moco')
    bmode(data=dataDict[f'Slice{ss}_data'][:,:,np.newaxis,:],
            plot=True,
            ID=f'Slice{ss}_bmode',
            path=img_save_dir)
#MP02.bmode(data=dataDict1[f'Slice{ss}_data'][:,:,np.newaxis,:])
#%%
def plotAve(data,masklv,plot=False,path=None,ID=None):
    data=data.squeeze()
    masklv=masklv.squeeze()
    indList=[]
    for i in range(np.shape(masklv)[-1]):
        if np.sum(masklv[:,:,i])==0:
            continue
        else:
            indList.append(i)
    Nx,Ny,Nd=np.shape(data.squeeze())
    dataNew=np.zeros((Nx,Ny))
    dataNew = np.mean(data[...,indList],axis=-1)
    plt.imshow(dataNew,cmap='gray')
    plt.axis('off')
    if plot==True:
        dir=os.path.join(path,ID)
        plt.savefig(dir)

    plt.show()
for ss in range(3):
    print('raw')
    plotAve(data=dataDict[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],
            masklv=dataDict[f'Slice{ss}_lv_raw'][:,:,np.newaxis,:],
            plot=True,
            ID=f'Slice{ss}_mean_raw',
            path=img_save_dir)
    print('moco')
    plotAve(data=dataDict[f'Slice{ss}_data'][:,:,np.newaxis,:],
            masklv=dataDict[f'Slice{ss}_lv'][:,:,np.newaxis,:], 
            plot=True,
            ID=f'Slice{ss}_mean',
            path=img_save_dir)




#%%

%matplotlib inline
plt.figure()

for ss in range(3):
    plt.subplot(3,2,2*ss+1)

    image_data=dataDict[f'Slice{ss}_data']
    mask_lv=dataDict[f'Slice{ss}_lv']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')

for ss in range(3):
    plt.subplot(3,2,2*ss+2)
    image_data=dataDict[f'Slice{ss}_data_raw']
    mask_lv=dataDict[f'Slice{ss}_lv_raw']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')
plt.savefig(os.path.join(img_save_dir,f'contours'))
plt.show()

#%%
###############Reseg
ss=1
image_data=dataDict[f'Slice{ss}_data']
mask_lv=dataDict[f'Slice{ss}_lv']
mask_endo=dataDict[f'Slice{ss}_endo']
mask_epi=dataDict[f'Slice{ss}_epi']
%matplotlib inline

###############Change the mask here
plot_ROI(image_data,mask_endo)
#%%
%matplotlib qt
d=[5]
mask_endo_tmp,mask_epi_tmp,mask_lv_tmp=draw_ROI_all(image_data,brightness=brightness,cmap=cmap,d=d)
for ind,dd in enumerate(d):
    mask_endo[:,:,dd]=mask_endo_tmp[:,:,ind]
    mask_epi[:,:,dd]=mask_epi_tmp[:,:,ind]
    mask_lv[:,:,dd]=mask_lv_tmp[:,:,ind]


#%%
#Save again
#%%

saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
with open(saveDict_name, 'wb') as fp:
    pickle.dump(dataDict, fp)
    print('dictionary saved successfully to file')



# %%
#Test load
saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
with open(saveDict_name, 'rb') as inp:
    dataDict2 = pickle.load(inp)
print('dictionary Load successfully to file')
#%%
for key,val in dataDict2.items():
    print(f'{key}:{np.shape(val)}')



for ss in range(3):
    plt.subplot(3,2,2*ss+1)

    image_data=dataDict2[f'Slice{ss}_data']
    mask_lv=dataDict2[f'Slice{ss}_lv']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')

for ss in range(3):
    plt.subplot(3,2,2*ss+2)
    image_data=dataDict2[f'Slice{ss}_data_raw']
    mask_lv=dataDict2[f'Slice{ss}_lv_raw']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')
plt.savefig(os.path.join(img_save_dir,f'contours'))
plt.show()

# %%
