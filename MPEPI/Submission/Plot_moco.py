#%%
%matplotlib inline
import os
import sys
sys.path.append('../') 
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import readFolder,decompose_LRT,moco,moco_naive
import numpy as np
import matplotlib.pyplot as plt
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

matplotlib.rcParams['savefig.dpi'] = 400

plot=False
# %%
CIRC_ID='CIRC_00446'
img_root_dir = os.path.join(defaultPath, "DataSet",CIRC_ID)
saved_img_root_dir=os.path.join(defaultPath, "DataSet",CIRC_ID,"save_imgs")
if not os.path.exists(saved_img_root_dir):
            os.mkdir(saved_img_root_dir)

# image root directory
# Statas saved 

#Read the MP01-MP03
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('.mapping'):
            if path.endswith('p.mapping')==False and path.endswith('m.mapping')==False:
                mapList.append(path)
print(mapList)
#%%
img_moco_dir='C:\Research\MRI\MP_EPI\Moco_Dec6\MOCO'
moco_data=h5py.File(os.path.join(img_moco_dir,rf'{CIRC_ID}_MOCO.mat'),'r')
#Create the GIF and see the images:

for ind,key in enumerate(moco_data):
    print(key,np.shape(moco_data[key]))
    try:
        data_tmp=np.transpose(moco_data[key],(2,1,0))
        img_path= os.path.join(saved_img_root_dir,f'Slice{ind-3}_Yuchi_moco_.gif')
        A2=np.copy(data_tmp)
        A2=data_tmp/np.max(data_tmp)*255
        mp_tmp=mapping(A2)
        mp_tmp.createGIF(img_path,A2,fps=5)
        #mp_tmp.imshow_corrected(volume=data_tmp[:,:,np.newaxis,:],valueList=range(1000),ID=f'Slice{ind-3}_Yuchi_moco',plot=plot,path=saved_img_root_dir)
        #mp_tmp.go_crop_Auto()
        #mp_tmp.go_resize()
        #mp_tmp.imshow_corrected(volume=data_tmp[:,:,np.newaxis,:],valueList=range(1000),ID=f'Slice{ind-3}_Yuchi_moco_cropped',plot=plot,path=saved_img_root_dir)
    except:

        pass



#%%
MP01_0=mapping(mapList[0])
MP01_1=mapping(mapList[1])
MP01_2=mapping(mapList[2])
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)

MP03=mapping(mapList[5])


#%%
MP01_list=[MP01_0,MP01_1,MP01_2]
MPs_list=[MP01,MP02,MP03]

for obj in MPs_list:
    obj._data=np.copy(obj._raw_data)
for obj in MP01_list:
    obj._data=np.copy(obj._raw_data)
#Get the shape of all data, and then replace the data with corrected
#Read the data
#Renew the dataset:
for ss,obj_T1 in enumerate(MP01_list):
    key=f'moco_Slice{ss}'
    moco_data_single_slice=np.transpose(moco_data[key],(2,1,0))
    Ndtmp=0
    for obj in MPs_list:
        if 'mp01' in obj.ID.lower():
            Ndtmp_end=np.shape(MP01_list[ss]._data)[-1]
            obj_T1._update_data(moco_data_single_slice[:,:,np.newaxis,0:Ndtmp_end])
            #obj_T1.go_crop_Auto()
            #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped',plot=plot,path=saved_img_root_dir)
            #obj_T1.go_create_GIF(path_dir=str(saved_img_root_dir))
            #obj_T1._update_data(moco_data_single_slice[:,:,0:Ndtmp_end])
            print(obj_T1.ID,np.shape(obj_T1._data))
            print('valueList=',obj_T1.valueList)
        else:
            Ndtmp_start=Ndtmp_end
            Ndtmp_end+=np.shape(obj._data)[-1]
            obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
            print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
            #print('valueList=',obj.valueList)

#%%
#Update Mask
#Read the contour:
MP02_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP02_T2_p.mapping'))
#Read the contour to others
#Read the mask

#%%
%matplotlib qt
#Crop the data
MP02_p.go_crop()
cropzone=MP02_p.cropzone
#%%

temp=np.shape(MP02_p._data)
shape=(temp[0], temp[1], MP02_p.shape[2])
mask_lv_crop=np.zeros(shape)
mask_endo_crop=np.zeros(shape)
mask_epi_crop=np.zeros(shape)
CoM_crop=np.zeros(shape)
for z in range(MP02_p.Nz):
    mask_lv_crop[:,:,z] = imcrop(MP02_p.mask_lv[:,:,z], cropzone)
    mask_endo_crop[:,:,z] = imcrop(MP02_p.mask_endo[:,:,z], cropzone)
    mask_epi_crop[:,:,z]= imcrop(MP02_p.mask_epi[:,:,z], cropzone)
    #CoM_crop=imcrop(MP02_p.CoM[:,:,z], cropzone)
    
#%%
import cv2 as cv

image=mask_lv_crop[:,:,0]*255-1
img=np.float16(image)
from skimage import measure
contours = measure.find_contours(img, 240)

# Display the image and plot all contours found
fig, ax = plt.subplots()
#ax.imshow(img, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# %%
#Draw the overlay before and after moco
#Before the moco: use _raw_data
#Only needs the first slice:
fileList=MPs_list+MP01_list
for obj in fileList:
    obj.cropzone=cropzone
    obj.go_crop()

#%%
#Only show 3 MP, and 6 images
raw_volume=np.zeros((temp[0], temp[1],3, 6))
#MP01:
raw_volume[:,:,0,:]=MP01_0._data[:,:,0,[0,4,7,10,12,14]]
raw_volume[:,:,1,:]=MP02._data[:,:,0,[0,2,3,4,6,7]]
raw_volume[:,:,2,:]=MP03._data[:,:,0,range(6)]


#%%
Nx,Ny,Nz,Nd=np.shape(raw_volume)
plt.style.use('dark_background')
cmap='gray'
valueList1=[MP01_0.valueList[i] for i in [0,4,7,10,12,14]]
valueList2=[MP02.valueList[i] for i in [0,2,3,4,6,7]]
valueList3=['50x','50y','50z','500x','500y','500z']
valueList=[valueList1,valueList2,valueList3]

#%%
#mask_lv_crop_tmp=mask_lv_crop[:,:,0]*255
#mask_lv_crop_tmp = np.array(mask_lv_crop[:,:,0], np.uint8)
#cnts = cv2.findContours(mask_lv_crop_tmp[..., 0], mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
#%%
fig,axs=plt.subplots(Nz,Nd, figsize=(Nd*3.3,Nz*3),constrained_layout=True)            
for d in range(Nd):
    for z in range(Nz):
        vmin = np.min(raw_volume[:,:,z,:])
        vmax = np.max(raw_volume[:,:,z,:])
        #alpha = mask_lv_crop[..., 0] * 1.0
        axs[z,d].imshow(raw_volume[:,:,z,d],cmap=cmap,vmin=vmin,vmax=vmax)
        masked_mask = np.ma.masked_where(mask_lv_crop[..., 0] == 1, mask_lv_crop[..., 0])
        axs[z,d].contour(masked_mask, colors='red') 
        
        #axs[z,d].imshow(canvas)
        #axs[z,d].imshow(mask_epi_crop[:,:,0])
        #axs[z,d].imshow(mask_endo_crop[:,:,0])
        axs[z,d].set_title(f'{valueList[z][d]}',fontsize='small')
        axs[z,d].axis('off')
        for contour in contours:
            axs[z,d].plot(contour[:, 1], contour[:, 0], linewidth=0.3)

plt.show()
#root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
img_dir= os.path.join(saved_img_root_dir,f'Mosaic')
#img_dir= os.path.join(saved_img_root_dir,f'Mosaic_contour')

if plot:
    plt.savefig(img_dir, bbox_inches='tight')
    plt.savefig(img_dir+'.pdf',bbox_inches='tight')



#%%
#Show the maps:



# %%
########Raw below
MP01_list=[MP01_0,MP01_1,MP01_2]
MPs_list=[MP01,MP02,MP03]

for obj in MPs_list:
    obj._data=np.copy(obj._raw_data)
for obj in MP01_list:
    obj._data=np.copy(obj._raw_data)

fileList=MPs_list+MP01_list
for obj in fileList:
    obj.cropzone=cropzone
    obj.go_crop()
#%%
#Only show 3 MP, and 6 images
#Now it's the raw data
raw_volume=np.zeros((temp[0], temp[1],3, 6))
#MP01:
raw_volume[:,:,0,:]=MP01_0._data[:,:,0,[0,4,7,10,12,14]]
raw_volume[:,:,1,:]=MP02._data[:,:,0,[0,2,3,4,6,7]]
raw_volume[:,:,2,:]=MP03._data[:,:,0,range(6)]


fig,axs=plt.subplots(Nz,Nd, figsize=(Nd*3.3,Nz*3),constrained_layout=True)            
for d in range(Nd):
    for z in range(Nz):
        vmin = np.min(raw_volume[:,:,z,:])
        vmax = np.max(raw_volume[:,:,z,:])
        #alpha = mask_lv_crop[..., 0] * 1.0
        axs[z,d].imshow(raw_volume[:,:,z,d],cmap=cmap,vmin=vmin,vmax=vmax)
        masked_mask = np.ma.masked_where(mask_lv_crop[..., 0] == 1, mask_lv_crop[..., 0])
        axs[z,d].contour(masked_mask, colors='red') 
        
        #axs[z,d].imshow(canvas)
        #axs[z,d].imshow(mask_epi_crop[:,:,0])
        #axs[z,d].imshow(mask_endo_crop[:,:,0])
        axs[z,d].set_title(f'{valueList[z][d]}',fontsize='small')
        axs[z,d].axis('off')
        for contour in contours:
            axs[z,d].plot(contour[:, 1], contour[:, 0], linewidth=0.3)

plt.show()
#root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'

img_dir= os.path.join(saved_img_root_dir,f'Mosaic_raw_contour')
if plot:
    plt.savefig(img_dir, bbox_inches='tight')
    plt.savefig(img_dir+'.pdf',bbox_inches='tight')
                
# %%
#Finally plot the mask/maps
MP01_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP01_T1_p.mapping'))
MP02_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP02_T2_p.mapping'))
MP03_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP03_DWI_p.mapping'))

temp=np.shape(MP02._data)
shape=(temp[0], temp[1], MP02.shape[2])
raw_volume=np.zeros((temp[0], temp[1],3))
#MP01:

raw_volume[:,:,0]=imcrop(MP01_p._map[:,:,0], cropzone)
raw_volume[:,:,1]=imcrop(MP02_p._map[:,:,0], cropzone)
raw_volume[:,:,2]=imcrop(MP03_p._map[:,:,0], cropzone)
#%%
#DRAW THE MAPS
fig,axs=plt.subplots(3,1, figsize=(1*3.3,3.3*3),constrained_layout=True)            

#alpha = mask_lv_crop[..., 0] * 1.0

im1=axs[0].imshow(raw_volume[:,:,0], cmap='magma',vmin=0,vmax=3000)
im2=axs[1].imshow(raw_volume[:,:,1],cmap='viridis',vmin=0,vmax=150)
im3=axs[2].imshow(raw_volume[:,:,2], cmap='hot',vmin=0,vmax=3)
#fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.018, aspect=11)
#fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.018, aspect=11)
#fig.colorbar(im3, ax=axs[2], shrink=0.7, pad=0.018, aspect=11)
#axs[z,d].imshow(canvas)
#axs[z,d].imshow(mask_epi_crop[:,:,0])
#axs[z,d].imshow(mask_endo_crop[:,:,0])
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()

img_dir= os.path.join(saved_img_root_dir,f'Maps_Slice0_2')
if plot:
    plt.savefig(img_dir, bbox_inches='tight')
    plt.savefig(img_dir+'.pdf',bbox_inches='tight')
                
# %%
           
MP02_p.cropzone=cropzone
MP03_p.cropzone=cropzone
MP02_p.go_crop()
MP03_p.go_crop()
#%%
fig,axs=plt.subplots(3,1, figsize=(1*3.3,3.3*3),constrained_layout=True) 
#alpha = mask_lv_crop[..., 0] * 1.0
base_im_tmp=np.array([MP02_p._data[:, :, 0, 0],MP03_p._data[:, :, 0, 0]])
base_im = np.mean(base_im_tmp,axis=0)
brightness = 0.8
alpha = mask_lv_crop[..., 0] * 1.0
axs[0].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
axs[1].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
axs[2].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
im1=axs[0].imshow(raw_volume[:,:,0], alpha=alpha,cmap='magma',vmin=0,vmax=3000)
im2=axs[1].imshow(raw_volume[:,:,1], alpha=alpha,cmap='viridis',vmin=0,vmax=150)
im3=axs[2].imshow(raw_volume[:,:,2], alpha=alpha,cmap='hot',vmin=0,vmax=3)
fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.018, aspect=11)
fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.018, aspect=11)
fig.colorbar(im3, ax=axs[2], shrink=0.7, pad=0.018, aspect=11)
#axs[z,d].imshow(canvas)
#axs[z,d].imshow(mask_epi_crop[:,:,0])
#axs[z,d].imshow(mask_endo_crop[:,:,0])
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()

img_dir= os.path.join(saved_img_root_dir,f'Maps_Slice0_overlay')
if plot:
    plt.savefig(img_dir, bbox_inches='tight')
    plt.savefig(img_dir+'.pdf',bbox_inches='tight')
                
# %%
#%%
#Plot un corrected
img_moco_dir='C:\Research\MRI\MP_EPI\Moco_Dec6'
moco_data=sio.loadmat(os.path.join(img_moco_dir,rf'{CIRC_ID}.mat'))
#Create the GIF and see the images:
uncorrected_list=[]
for z in range(3):
    data=np.concatenate((moco_data[f'MP01_Slice{z}'].squeeze(),moco_data[f'MP02_Slice{z}'].squeeze(),moco_data[f'MP03_Slice{z}'].squeeze()),axis=-1)
    img_path= os.path.join(saved_img_root_dir,f'Slice{z+1}.gif')
    A2=np.copy(data)
    A2=data/np.max(data)*255
    mp_tmp=mapping(A2)
    mp_tmp.createGIF(img_path,A2,fps=5)
    uncorrected_list.append(data)
#%%

#%%
MP01_0=mapping(mapList[0])
MP01_1=mapping(mapList[1])
MP01_2=mapping(mapList[2])
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)

MP03=mapping(mapList[5])


#%%
MP01_list=[MP01_0,MP01_1,MP01_2]
MPs_list=[MP01,MP02,MP03]

for obj in MPs_list:
    obj._data=np.copy(obj._raw_data)
for obj in MP01_list:
    obj._data=np.copy(obj._raw_data)
#%%
#Get the shape of all data, and then replace the data with corrected
#Read the data
#Renew the dataset:
for ss,obj_T1 in enumerate(MP01_list):
    #key=f'moco_Slice{ss}'
    moco_data_single_slice=uncorrected_list[ss]
    Ndtmp=0
    for obj in MPs_list:
        if 'mp01' in obj.ID.lower():
            Ndtmp_end=np.shape(MP01_list[ss]._data)[-1]
            obj_T1._update_data(moco_data_single_slice[:,:,np.newaxis,0:Ndtmp_end])
            #obj_T1.go_crop_Auto()
            #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped',plot=plot,path=saved_img_root_dir)
            #obj_T1.go_create_GIF(path_dir=str(saved_img_root_dir))
            #obj_T1._update_data(moco_data_single_slice[:,:,0:Ndtmp_end])
            print(obj_T1.ID,np.shape(obj_T1._data))
            print('valueList=',obj_T1.valueList)
        else:
            Ndtmp_start=Ndtmp_end
            Ndtmp_end+=np.shape(obj._data)[-1]
            obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
            print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
            #print('valueList=',obj.valueList)

# %%
%matplotlib qt
#Crop both:
MP02.go_crop()
cropzone=MP02.cropzone
#MP02.go_resize()
for obj in [MP01,MP03]:
    obj.cropzone=cropzone
    obj.go_crop()
    #obj.go_resize()

# %%

#%%
MP03.go_cal_ADC()
MP03.imshow_map(path=img_root_dir,plot=plot)
MP03.save(filename=os.path.join(img_root_dir,f'{MP03.ID}_uncorrected_m.mapping'))

#%%
MP02.go_t2_fit()
MP02.imshow_map(path=img_root_dir,plot=plot)
MP02.save(filename=os.path.join(img_root_dir,f'{MP02.ID}_uncorrected_m.mapping'))

#%%
#---...-----
#Please calculate the maps in a loop
MP01._map=np.copy(MP02._map)
for ind,mp in enumerate(MP01_list):
    mp.cropzone=cropzone
    mp.go_crop()
    mp._delete(d=-1)
    finalMap,finalRa,finalRb,finalRes=mp.go_ir_fit(searchtype='grid')
    MP01._map[:,:,ind]=finalMap.squeeze()
MP01.imshow_map(path=img_root_dir,plot=plot)
MP01.save(filename=os.path.join(img_root_dir,f'{MP01.ID}_uncorrected_m.mapping'))

# %%
plt.style.use('default')
MP01.imshow_map(path=img_root_dir,plot=True)
MP03.imshow_map(path=img_root_dir,plot=True)
MP02.imshow_map(path=img_root_dir,plot=True)

# %%
