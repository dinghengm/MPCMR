####I don't remember the path
######It seems like copy all to DATASET
######Plot data
######
#%%
%matplotlib inline
import os
import sys
sys.path.append('../') 
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import *
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
from SimulationFunction import *
matplotlib.rcParams['savefig.dpi'] = 400

plot=True
# %%
CIRC_ID='CIRC_00446'
img_root_dir = os.path.join(defaultPath, "DataSet",CIRC_ID)
##Feb22#Save_imgs_2
saved_img_root_dir=os.path.join(defaultPath, "DataSet",CIRC_ID,"save_imgs_resize")

#saved_img_root_dir=os.path.join(defaultPath, "DataSet",CIRC_ID,"save_imgs")
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
        #mp_tmp.createGIF(img_path,A2,fps=5)
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

MP01=mapping(mapList[-3])
MP02=mapping(mapList[-2])
MP03=mapping(mapList[-1])
'''
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
MP03=mapping(mapList[5])
'''



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
MP02_p=mapping(mapList[-2])
MP01=mapping(mapList[-3])
MP02=mapping(mapList[-2])
MP03=mapping(mapList[-1])
#Read the contour to others
#Read the mask
cropzone=MP02_p.cropzone
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
    mask_lv_crop[:,:,z] = imresize(imcrop(MP02_p.mask_lv[:,:,z], cropzone),(temp[0], temp[1]))
    mask_endo_crop[:,:,z] = imresize(imcrop(MP02_p.mask_endo[:,:,z], cropzone),(temp[0], temp[1]))
    mask_epi_crop[:,:,z]= imresize(imcrop(MP02_p.mask_epi[:,:,z], cropzone),(temp[0], temp[1]))
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
import copy
MP01_p=copy.copy(MP01)
MP02_p=copy.copy(MP02)
MP03_p=copy.copy(MP03)

fileList=MP01_list
for obj in fileList:
    obj.cropzone=cropzone
    obj.go_crop()
    obj.go_resize()
#%%
#Only show 3 MP, and 6 images

raw_volume=np.zeros((temp[0], temp[1],3, 6))
#MP01:
raw_volume[:,:,0,:]=MP01_0._data[:,:,0,[0,4,7,10,12,14]]
raw_volume[:,:,1,:]=MP02._data[:,:,0,[0,2,3,4,6,7]]
raw_volume[:,:,2,:]=MP03._data[:,:,0,range(6)]


#%%
Nx,Ny,Nz,Nd=np.shape(raw_volume)
#Feb22
plt.style.use('dark_background')
#plt.style.use('dark_background')
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
#Single Nd
for z in range(3):
    fig,axs=plt.subplots(1,Nd, figsize=(Nd*3.3,Nz*3),constrained_layout=True)            
    for d in range(Nd):
        
            vmin = np.min(raw_volume[:,:,z,:])
            vmax = np.max(raw_volume[:,:,z,:])
            #alpha = mask_lv_crop[..., 0] * 1.0
            axs[d].imshow(raw_volume[:,:,z,d],cmap=cmap,vmin=vmin,vmax=vmax)
            #masked_mask = np.ma.masked_where(mask_lv_crop[..., 0] == 1, mask_lv_crop[..., 0])
            #axs[d].contour(masked_mask, colors='red') 
            
            #axs[z,d].imshow(canvas)
            #axs[z,d].imshow(mask_epi_crop[:,:,0])
            #axs[z,d].imshow(mask_endo_crop[:,:,0])
            #axs[d].set_title(f'{valueList[z][d]}',fontsize='small')
            axs[d].axis('off')
            #for contour in contours:
            #    axs[d].plot(contour[:, 1], contour[:, 0], linewidth=0.3)

    #root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
    img_dir= os.path.join(saved_img_root_dir,f'Mosaic{z}')
    #img_dir= os.path.join(saved_img_root_dir,f'Mosaic_contour')

    if plot:
        plt.savefig(img_dir, bbox_inches='tight')
        plt.savefig(img_dir+'.pdf',bbox_inches='tight')
    
    plt.show()
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
'''MP01_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP01_T1_p.mapping'))
MP02_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP02_T2_p.mapping'))
MP03_p=mapping(data=os.path.join(img_root_dir,'CIRC_00446_MP03_DWI_p.mapping'))
'''



temp=np.shape(MP02._data)
shape=(temp[0], temp[1], MP02.shape[2])
raw_volume=np.zeros((temp[0], temp[1],3))
#MP01:
'''
raw_volume[:,:,0]=imcrop(MP01_p._map[:,:,0], cropzone)
raw_volume[:,:,1]=imcrop(MP02_p._map[:,:,0], cropzone)
raw_volume[:,:,2]=imcrop(MP03_p._map[:,:,0], cropzone)

'''
#%%
#DRAW THE MAPS
temp=np.shape(MP02._data)
shape=(temp[0], temp[1], MP02.shape[2])
raw_volume=np.zeros((temp[0], temp[1],3))
raw_volume[:,:,0]=MP01._map[:,:,0]
raw_volume[:,:,1]=MP02._map[:,:,0]
raw_volume[:,:,2]=MP03._map[:,:,0]
fig,axs=plt.subplots(3,1, figsize=(1*3.3,3.3*3),constrained_layout=True)            

#alpha = mask_lv_crop[..., 0] * 1.0

im1=axs[0].imshow(raw_volume[:,:,0], cmap='magma',vmin=0,vmax=3000)
im2=axs[1].imshow(raw_volume[:,:,1],cmap='viridis',vmin=0,vmax=150)
im3=axs[2].imshow(raw_volume[:,:,2], cmap='hot',vmin=0,vmax=3)
fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.018, aspect=11)
fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.018, aspect=11)
fig.colorbar(im3, ax=axs[2], shrink=0.7, pad=0.018, aspect=11)
#axs[0].imshow(canvas)
#axs[0].imshow(mask_epi_crop[:,:,0])
#axs[0].imshow(mask_endo_crop[:,:,0])
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')


img_dir= os.path.join(saved_img_root_dir,f'Maps_Slice0_2')
if plot:
    plt.savefig(img_dir, bbox_inches='tight')
    plt.savefig(img_dir+'.pdf',bbox_inches='tight')
plt.show()          
# %%
           
MP02_p.cropzone=cropzone
MP03_p.cropzone=cropzone
MP02_p.go_crop()
MP03_p.go_crop()
#%%
plt.style.use('default')
fig,axs=plt.subplots(3,1, figsize=(1*3.3,3.3*3),constrained_layout=True) 
#alpha = mask_lv_crop[..., 0] * 1.0
base_im_tmp=np.array([MP02_p._data[:, :, 0, 0],MP03_p._data[:, :, 0, 0]])
base_im = np.mean(base_im_tmp,axis=0)
brightness = 0.8
#alpha = 1.0*MP02_p.mask_lv[..., 0]
axs[0].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
axs[1].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
axs[2].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
im1=axs[0].imshow(raw_volume[:,:,0],cmap='magma',vmin=0,vmax=3000, alpha=1.0*MP02_p.mask_lv[..., 0])
im2=axs[1].imshow(raw_volume[:,:,1],cmap='viridis',vmin=0,vmax=150, alpha=1.0*MP02_p.mask_lv[..., 0])
im3=axs[2].imshow(raw_volume[:,:,2],cmap='hot',vmin=0,vmax=3, alpha=1.0*MP02_p.mask_lv[..., 0])
#bar1=fig.colorbar(im1, ax=axs[0], shrink=0.7, pad=0.018, aspect=11)
#bar2=fig.colorbar(im2, ax=axs[1], shrink=0.7, pad=0.018, aspect=11)
#bar3=fig.colorbar(im3, ax=axs[2], shrink=0.7, pad=0.018, aspect=11)
#axs[z,d].imshow(canvas)
#axs[z,d].imshow(mask_epi_crop[:,:,0])
#axs[z,d].imshow(mask_endo_crop[:,:,0])
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')


img_dir= os.path.join(saved_img_root_dir,f'Maps_Slice0_overlay')
if plot:
    plt.savefig(img_dir, bbox_inches='tight')
    plt.savefig(img_dir+'.pdf',bbox_inches='tight')
plt.show()                
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
def ir_fit(data=None,TIlist=None,ra=500,rb=-1000,T1=600,type='WLS',error='l2',Niter=2,searchtype='grid',
            T1bound=[1,5000],invertPoint=4):
    aEstTmps=[]
    bEstTmps=[]
    T1EstTmps=[]
    resTmps=[]
    #Make sure the data come in increasing TI-order
    #index = np.argsort(TIlist)
    #ydata=np.squeeze(data[index])
    #Initialize variables:
    minIndTmps=[]
    minInd=np.argmin(data)
    '''if minInd==0:
        minInd=1
    elif minInd==len(TIlist):
        minInd=len(TIlist)-1'''
    #Invert the data to 2x*before,at, 2x*after the min
    invertPoint==None
    if invertPoint==None:
        iterNum=0,2
    else:
        iterNum=1-int(invertPoint/2),1+int(invertPoint/2)+1

    for ii in range(iterNum[0],iterNum[1],1):
        try:
            minIndTmp=minInd+int(ii)
            invertMatrix=np.concatenate((-np.ones(minIndTmp),np.ones(len(TIlist)-minIndTmp)),axis=0)
            dataTmp=data*invertMatrix.T
            minIndTmps.append(minIndTmp)
        except:
            continue

        if searchtype == 'lm':
            try: 
                T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_lm(data=dataTmp,TIlist=TIlist,
                ra=ra,rb=rb,T1=T1,type=type,error=error,Niter=Niter)
                aEstTmps.append(ra_exp)
                bEstTmps.append(rb_exp)
                T1EstTmps.append(T1_exp)
                #Get the chisquare
                resTmps.append(res)
            except:
                T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(data=dataTmp,TIlist=TIlist,T1bound=T1bound)
                aEstTmps.append(ra_exp)
                bEstTmps.append(rb_exp)
                T1EstTmps.append(T1_exp)
                #Get the chisquare
                resTmps.append(res)
        elif searchtype== 'grid':
            T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(data=dataTmp,TIlist=TIlist,T1bound=T1bound)
            aEstTmps.append(ra_exp)
            bEstTmps.append(rb_exp)
            T1EstTmps.append(T1_exp)
            #Get the chisquare
            resTmps.append(res)
    returnInd = np.argmin(np.array(resTmps))
    T1_final=T1EstTmps[returnInd]
    ra_final=aEstTmps[returnInd]
    rb_final=bEstTmps[returnInd]
    #ydata_exp=ir_recovery(TIlist,T1,ra,rb)
    return T1_final,ra_final,rb_final,resTmps,minIndTmps[returnInd]
#%%

pltDot=np.array([40,50])
base_im_tmp=np.array([MP02_p._data[:, :, 0, 0],MP03_p._data[:, :, 0, 0]])
base_im = np.mean(base_im_tmp,axis=0)
plt.imshow(base_im,cmap='gray')
plt.scatter(x=pltDot[1],y=pltDot[0])
plt.show()


###Finally we plot one line data and plot as a xyz
T1_line=MP01_0._data[pltDot[0],pltDot[1],0,:]   #First
T1_line=np.delete(T1_line,-1)

TIlist=np.array(MP01_0.valueList)
TIlist=np.delete(TIlist,-1).tolist()
print(TIlist,':',T1_line)
T1_final_grid,ra_final_grid,rb_final_grid,_,returnInd=ir_fit(abs(T1_line),TIlist,searchtype='grid',T1bound=[1,5000])

x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
ydata_exp=abs(ir_recovery(x_plot,T1_final_grid,ra_final_grid,rb_final_grid))
plt.plot(x_plot,ydata_exp)

plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(T1_line[0:returnInd]),color='r')
plt.scatter(np.array(TIlist),np.abs(T1_line))
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
#plt.title(f'Simulation T1={T1} SNR={SNR}')
#plt.axis(xmin=np.min(T1_line),xmax=np.max(T1_line),ymin=-1,ymax=1)
plt.text(x=4000,y=0,s=f'T1={int(T1_final_grid)}\ny={ra_final_grid:.02f}+{rb_final_grid:.02f}*e^-t/{int(T1_final_grid)}')
plt.grid('True')
plt.show()
plt.close()

#%%

zz=2
%matplotlib inline
pltDot=np.array([38,44])
plt.imshow(MP01._map[:,:,zz],cmap='magma',vmin=0,vmax=3000)
plt.scatter(x=pltDot[1],y=pltDot[0])
plt.show()

#%%
%matplotlib inline
mapT1=MP01_list[zz]
Nd=MP01_1.Nd
fig,axs=plt.subplots(1,Nd, figsize=(Nd*3.3,1*3),constrained_layout=True)            
for d in range(Nd):

        vmin = np.min(mapT1._data[:,:,0,:])
        vmax = np.max(mapT1._data[:,:,0,:])
        #alpha = mask_lv_crop[..., 0] * 1.0
        axs[d].imshow(mapT1._data[:,:,0,d],cmap='gray',vmin=vmin,vmax=vmax)
        #masked_mask = np.ma.masked_where(mask_lv_crop[..., 0] == 1, mask_lv_crop[..., 0])
        #axs[1,d].contour(masked_mask, colors='red') 
        
        #axs[z,d].imshow(canvas)
        #axs[z,d].imshow(mask_epi_crop[:,:,0])
        #axs[z,d].imshow(mask_endo_crop[:,:,0])
        axs[d].set_title(f'{mapT1.valueList[d]}',fontsize='small')
        axs[d].axis('off')
plt.show()
#Slice1

base_im_tmp=np.array([MP02_p._data[:, :, zz, 0],MP02_p._data[:, :, zz, 2]])
base_im = np.mean(base_im_tmp,axis=0)
plt.imshow(base_im,cmap='gray')
plt.scatter(x=pltDot[1],y=pltDot[0])
plt.show()


###Finally we plot one line data and plot as a xyz
T1_line=mapT1._data[pltDot[0],pltDot[1],0,:]   #First
T1_line=np.delete(T1_line,[-1,-6])
#T1_line=np.delete(T1_line,[-1])
TIlist=np.array(mapT1.valueList)
TIlist=np.delete(TIlist,[-1,-6]).tolist()
#TIlist=np.delete(TIlist,[-1]).tolist()
print(TIlist,':',T1_line)
T1_final_grid,ra_final_grid,rb_final_grid,_,returnInd=ir_fit(abs(T1_line),TIlist,searchtype='grid',T1bound=[1,5000])

x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
ydata_exp=abs(ir_recovery(x_plot,T1_final_grid,ra_final_grid,rb_final_grid))
plt.plot(x_plot,ydata_exp)

#plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(T1_line[0:returnInd]),color='r')
plt.scatter(np.array(TIlist),np.abs(T1_line))
plt.legend([f'T1={int(T1_final_grid)}'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
#plt.title(f'Simulation T1={T1} SNR={SNR}')
#plt.axis(xmin=np.min(T1_line),xmax=np.max(T1_line),ymin=-1,ymax=1)
#plt.text(x=4000,y=0,s=f'T1={int(T1_final_grid)}\ny={ra_final_grid:.02f}+{rb_final_grid:.02f}*e^-t/{int(T1_final_grid)}\nT1_map={mapT1._map[pltDot[0],pltDot[1],zz]}')
#print(f'1={int(T1_final_grid)}\ny={ra_final_grid:.02f}+{rb_final_grid:.02f}*e^-t/{int(T1_final_grid)}')
#print(f'T1_map={MP01._map[pltDot[0],pltDot[1],zz]}')
#plt.grid('True')
plt.show()
plt.close()










#%%
T2_line=MP02._data[pltDot[0],pltDot[1],2,:]
T2list=MP02.valueList
T2_exp,Mz_exp,res,ydata_exp=sub_mono_t2_fit_exp(T2_line,T2list)
T2=MP02._map[pltDot[0],pltDot[1],zz]
start=0
stop=150
x_plot=np.arange(start=start,stop=stop,step=1)
ydata_exp=abs(T2_recovery(x_plot,T2_exp,Mz_exp))*np.max(T2_line)

plt.plot(x_plot,ydata_exp)

#plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
plt.scatter(np.array(T2list),T2_line)
plt.legend([f'T2={T2_exp:.01f}'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
#plt.title(f'Simulation T2={T2}')
plt.axis(xmin=0,xmax=stop,ymin=0)
#plt.text(x=8000,y=0.5,s=f'T2={int(T2_exp)}\ny={Mz_exp:.02f}*e^-t/{T2_exp:.01f}')
#plt.grid('True')
plt.show()
plt.close()

#%%

zz=2
%matplotlib inline
pltDot=np.array([38,44])
plt.imshow(MP03._map[:,:,zz],cmap='hot',vmin=0,vmax=3)
plt.scatter(x=pltDot[1],y=pltDot[0])
plt.show()





#%%
Nd=MP03.Nd
adc_val=MP03._map[pltDot[0],pltDot[1],zz]
ADC_line=MP03._data[pltDot[0],pltDot[1],zz,:]
Nbval=np.shape(MP03.bval[MP03.bval==50])[0]
S50=np.zeros(Nbval)
S500=np.zeros(Nbval)
ADC_temp=np.zeros(Nbval)
for j in range(Nbval):
    S50[j]=ADC_line[j]
    ind=np.arange(j,Nd,Nbval)[1::]
    print('The b500 index:',ind)
    #Go through all the b500
    #Averaging all b500
    S500[j]=np.mean(ADC_line[ind])
ADC_temp=-1/450 * np.log(S500/S50)
ADCmap=np.mean(ADC_temp)    #/1000

S_b0=np.mean(S500)/np.exp(-500*ADCmap)
def ADC_exp(b,adc,M0):
    return M0*np.exp(-b*adc)


ADC=ADCmap*1000    #*1000
start=0
stop=1000
x_plot=np.arange(start=start,stop=stop,step=0.1)
ydata_exp=abs(ADC_exp(x_plot,ADCmap,S_b0))

plt.plot(x_plot,ydata_exp)

#plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')

plt.scatter(np.repeat(50.0,len(S50)),S50)
plt.scatter(np.repeat(500.0,len(S500)),S500)
plt.legend([f'ADC={ADC:.02f}'])
plt.xlabel('b val (s/mm2)')
plt.ylabel('Magnetization')
#plt.title(f'Simulation T2={adc_val}')
plt.axis(xmin=0,xmax=stop,ymin=0)
#plt.text(x=8000,y=0.5,s=f'T2={int(T2_exp)}\ny={Mz_exp:.02f}*e^-t/{T2_exp:.01f}')
#plt.grid('True')
plt.show()
plt.close()

# %%
