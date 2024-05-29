#%%
%matplotlib inline
import os
import sys
sys.path.append('../') 
import numpy as np
from libMapping_v13 import *  # <--- this is all you need to do diffusion processing
import matplotlib.pyplot as plt
import os
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400

#%%
plot=True

#%%
img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'CIRC_00488_1')
saved_img_root_dir=os.path.join(defaultPath, "Presentation")
if not os.path.exists(saved_img_root_dir):
    os.mkdir(saved_img_root_dir)


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

# %%
#Get the gif for raw and moco
del MP01
MP01_0=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488_1\MP01_Slice0_m.mapping')
MP01_1=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488_1\MP01_Slice1_m.mapping')
MP01_2=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488_1\MP01_Slice2_m.mapping')

MP01=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488\CIRC_00488_MP01_T1_p.mapping')
#%%
for obj in [MP01_0,MP01_1,MP01_2]:
    obj.go_crop_Auto()
    obj.go_resize()
    obj.cropzone=MP02.cropzone
    obj.go_crop()
    obj._update()


# %%
mocoData0=np.concatenate((MP01_0._data[...,0,:],MP02._data[...,0,:],MP03._data[...,0,:]),axis=-1)
mocoData1=np.concatenate((MP01_1._data[...,0,:],MP02._data[...,1,:],MP03._data[...,1,:]),axis=-1)
mocoData2=np.concatenate((MP01_2._data[...,0,:],MP02._data[...,2,:],MP03._data[...,2,:]),axis=-1)

for i,data in enumerate([mocoData0,mocoData1,mocoData2]):
    A0 = data/np.max(data)*255
    MP02.createGIF(data=A0,path=str(saved_img_root_dir+rf'\\MOCO_Slice{i}'))

# %%
cropzone=MP02.cropzone
for obj in [MP01_0,MP01_1,MP01_2,MP02,MP03]:
    obj._data=obj._raw_data
    obj.go_crop_Auto()
    obj.go_resize()
    obj.cropzone=cropzone
    obj.go_crop()
    obj._update()


# %%
mocoData0_Raw=np.concatenate((MP01_0._data[...,0,:],MP02._data[...,0,:],MP03._data[...,0,:]),axis=-1)
mocoData1_Raw=np.concatenate((MP01_1._data[...,0,:],MP02._data[...,1,:],MP03._data[...,1,:]),axis=-1)
mocoData2_Raw=np.concatenate((MP01_2._data[...,0,:],MP02._data[...,2,:],MP03._data[...,2,:]),axis=-1)
#Make Raw
for i,data in enumerate([mocoData0_Raw,mocoData1_Raw,mocoData2_Raw]):
    A0 = data/np.max(data)*255
    MP02.createGIF(data=A0,path=str(saved_img_root_dir+rf'\\MOCO_Slice{i}_Raw'))

# %%
MP01_0.go_ir_fit(invertPoint=8)
MP01_1.go_ir_fit(invertPoint=8)
MP01_2.go_ir_fit(invertPoint=8)
MP02.go_t2_fit()
MP03.go_cal_ADC()
# %%
# %%
def bmode(data=None,ID=None,x=None,y=None,plot=False,path=None):

    Nx,Ny,Nz,Nd=np.shape(data)
    if x==None and y==None:
        if np.shape(data)[0]>np.shape(data)[1]:
            x=int(np.shape(data)[0]/2)
        else:
            y=int(np.shape(data)[1]/2)
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
            ax[1,i].axis('off')
            ax[2,i].axis('off')
    elif Nz==1:
        plt.subplot(121)
        plt.imshow(data.squeeze()[...,0],cmap='gray')
        plt.axis('off')
        if x is not None:
            plt.axhline(y=x, color='r', linestyle='-')
        if y is not None:
            plt.axvline(x=y, color='r', linestyle='-')
        plt.subplot(122)
        A3=np.squeeze(A2)
        plt.imshow(A3,cmap='gray')
        plt.axis('off')
    if plot==True:
        dir=os.path.join(path,ID)
        plt.savefig(dir)
    plt.show()
def plotAve(data,plot=False,path=None,ID=None):
    data=data.squeeze()
    Nx,Ny,Nd=np.shape(data.squeeze())
    dataNew=np.zeros((Nx,Ny))
    dataNew = np.mean(data,axis=-1)
    plt.imshow(dataNew,cmap='gray')
    plt.axis('off')
    if plot==True:
        dir=os.path.join(path,ID)
        plt.savefig(dir)

    plt.show()

# %%
for i,data in enumerate([mocoData0_Raw,mocoData1_Raw,mocoData2_Raw]):
    bmode(data[:,:,np.newaxis,:],ID=f'bmode_raw_{i}',path=saved_img_root_dir,plot=True)
    plotAve(data[:,:,np.newaxis,:],ID=f'ave_raw_{i}',path=saved_img_root_dir,plot=True)
# %%
for i,data in enumerate([mocoData0,mocoData1,mocoData2]):
    bmode(data[:,:,np.newaxis,:],ID=f'bmode_{i}',path=saved_img_root_dir,plot=True)
    plotAve(data[:,:,np.newaxis,:],ID=f'ave_{i}',path=saved_img_root_dir,plot=True)

# %%
for ss in range(MP02.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP02._map[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_raw_T2')
    
    plt.savefig(img_dir)
MP02.save(filename=os.path.join(saved_img_root_dir,f'{MP02.ID}_raw_m.mapping'))
map_data=np.copy(MP02._map)
map_data[:,:,0]=np.squeeze(MP01_0._map)
map_data[:,:,1]=np.squeeze(MP01_1._map)
map_data[:,:,2]=np.squeeze(MP01_2._map)
MP01._map= np.squeeze(map_data)
#%%
for ss,obj_T1 in enumerate([MP01_0,MP01_1,MP01_2]):
    plt.figure()
    plt.axis('off')
    plt.imshow(obj_T1._map.squeeze(),cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_raw_T1')
    plt.savefig(img_dir)
    obj_T1.save(filename=os.path.join(saved_img_root_dir,f'{obj_T1.ID}_raw_m.mapping'))

#%%
Nx,Ny,Nz,Nd=np.shape(MP03._data)
for ss in range(MP03.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP03.ADC[:,:,ss],cmap='hot',vmin=0,vmax=3)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_raw_DWI')
    plt.savefig(img_dir)
MP03.save(filename=os.path.join(saved_img_root_dir,f'{MP03.ID}_raw_m.mapping'))
MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.ID}_raw_m.mapping'))



#%%

#############MOCO below
# %%

MP01_moco=mapping(mapList[0])
MP02_moco=mapping(mapList[1])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03_moco=mapping(mapList[2])
#%%
newmap=np.zeros(np.shape(MP02._map))

for obj in [MP01_moco,MP02_moco,MP03_moco]:
    for z in range(3):
        newmap[:,:,z]=imcrop(MP01_moco._map[:,:,z], cropzone)
    MP01_moco._map=newmap

#%%

for ss in range(MP02.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP02_moco._map[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_T2')
    plt.savefig(img_dir)
MP02_moco.save(filename=os.path.join(saved_img_root_dir,f'{MP02.ID}_m.mapping'))

#%%
for ss in range(MP01.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP01_moco._map[:,:,ss],cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_T1')
    plt.savefig(img_dir)
MP01_moco.save(filename=os.path.join(saved_img_root_dir,f'{MP01.ID}_m.mapping'))

#%%
for ss in range(MP03.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP03_moco.ADC[:,:,ss],cmap='hot',vmin=0,vmax=3)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_DWI')
    plt.savefig(img_dir)
MP03_moco.save(filename=os.path.join(saved_img_root_dir,f'{MP03.ID}_m.mapping'))

# %%
#Get the gif for raw and moco
del MP01
MP01_0_moco=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488_1\MP01_Slice0_m.mapping')
MP01_1_moco=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488_1\MP01_Slice1_m.mapping')
MP01_2_moco=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488_1\MP01_Slice2_m.mapping')

MP01_moco=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488\CIRC_00488_MP01_T1_p.mapping')
#%%
for obj in [MP01_0,MP01_1,MP01_2]:
    obj.go_crop_Auto()
    obj.go_resize()
    obj.cropzone=MP02.cropzone
    obj.go_crop()
    obj._update()
