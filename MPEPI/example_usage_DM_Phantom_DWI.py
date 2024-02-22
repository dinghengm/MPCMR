# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
from roipoly import RoiPoly, MultiRoi
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 500
from t1_fitter import T1_fitter,go_fit_T1
plt.rcParams.update({'axes.titlesize': 'small'})
#%%
plot=True
crange=[0,2]
# %%



#CIRC_ID='CIRC_Phantom_Aug9'
CIRC_ID='CIRC_Phantom_Feb_12_Diff_SE_DWI'
#dicomPath=os.path.join(defaultPath,f'20230809_1449_CIRC_Phantom_Aug_9th_Diff_\MP03_DWI\MR000000.dcm')
#dirpath=os.path.dirname(dicomPath)
dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_PHANTOM_FEB12 CIRC_PHANTOM_FEB12\MR ep2d_MP03_DWI_Z_FINAL1'
MP03=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03.go_crop()
MP03.go_resize(scale=2)
cropzone=MP03.cropzone
MP03.imshow_corrected(ID='MP03_DWI',valueList=range(1000),plot=False)
#%%
plt.style.use('default')

MP03.go_cal_ADC()
MP03._map=MP03.ADC-0.3
MP03.imshow_map(plot=plot,crange=crange,cmap='hot')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
#%%
plt.figure(constrained_layout=True)
plt.imshow(MP03._map[:,:,1],vmin=crange[0],vmax=crange[1],cmap='hot')
plt.axis('off')
plt.savefig(os.path.join(os.path.dirname(MP03.path),'MP03'))
#%%
#SE
dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_PHANTOM_FEB12 CIRC_PHANTOM_FEB12\MR ep2d_MP03_DWI_Z_FINAL1_1ave'
MP03_EPI_Ave=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03_EPI_Ave.cropzone=cropzone
MP03_EPI_Ave.go_crop()
MP03_EPI_Ave.go_resize(scale=2)
fig,axs=MP03_EPI_Ave.imshow_corrected(ID='Reference_DWI',plot=plot,valueList=range(1000))
MP03_EPI_Ave.go_cal_ADC()
MP03_EPI_Ave._map=MP03_EPI_Ave.ADC-0.3
plt.style.use('default')
MP03_EPI_Ave.imshow_map(plot=plot,crange=crange,cmap='hot')
#%%
plt.figure(constrained_layout=True)
plt.imshow(MP03._map[:,:,1],vmin=crange[0],vmax=crange[1],cmap='hot')
plt.axis('off')
plt.savefig(os.path.join(os.path.dirname(MP03.path),'MP03_Ref'))
#%%
#Ave data
cmap='hot'
crange=[0.5,1.5]
image = MP03_EPI_Ave._map[:,:,2]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(2))
fig.canvas.manager.set_window_title('Slice '+ str(2))
multirois = MultiRoi(fig=fig, roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())
Nd=len(roi_names)


#%%
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')

path=os.path.dirname(MP03_EPI_Ave.path)
img_dir= os.path.join(path,f'{MP03_EPI_Ave.ID}')
np.save(img_dir+'_map',image)
np.save(img_dir+'_roi',roi_mask)

#%%
#Ave data
cmap='hot'
crange=[0.5,1.5]
image = MP03._map[:,:,2]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(2))
fig.canvas.manager.set_window_title('Slice '+ str(2))
multirois = MultiRoi(fig=fig, roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())
Nd=len(roi_names)


#%%
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')
path=os.path.dirname(MP03.path)
img_dir= os.path.join(path,f'{MP03.ID}')
#%%
np.save(img_dir+'_map',image)
np.save(img_dir+'_roi',roi_mask)
#%%
#CIRC_ID='CIRC_Phantom_Aug9'
CIRC_ID='CIRC_Phantom_Feb_12_Diff_SE_DWI'
#dicomPath=os.path.join(defaultPath,f'20230809_1449_CIRC_Phantom_Aug_9th_Diff_\MP03_DWI\MR000000.dcm')
#dirpath=os.path.dirname(dicomPath)
dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_PHANTOM_FEB12 CIRC_PHANTOM_FEB12\MR ep2d_MP03_DWI_Z_FINAL2'
MP03=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03.cropzone=cropzone
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.imshow_corrected(valueList=range(1000),plot=True)
#%%
plt.style.use('default')
MP03.go_cal_ADC()
MP03.imshow_map(plot=plot,crange=[0,2.5],cmap='hot')
# %%
#Ave data
cmap='hot'
crange=[0,2.5]
image = MP03._map[:,:,2]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(2))
fig.canvas.manager.set_window_title('Slice '+ str(2))
multirois = MultiRoi(fig=fig, roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())
Nd=len(roi_names)


#%%
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')
path=os.path.dirname(MP03.path)
img_dir= os.path.join(path,f'{MP03.ID}')
#%%
np.save(img_dir+'_map',image)
np.save(img_dir+'_roi',roi_mask)
#%%
CIRC_ID='CIRC_Phantom_Feb_12_Diff_SE_DWI'
#dicomPath=os.path.join(defaultPath,f'20230809_1449_CIRC_Phantom_Aug_9th_Diff_\MP03_DWI\MR000000.dcm')
#dirpath=os.path.dirname(dicomPath)
dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_PHANTOM_FEB12 CIRC_PHANTOM_FEB12\MR ep2d_MP03_DWI_Z_FINAL1'
MP03=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03.cropzone=cropzone
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.imshow_corrected(ID='MP03_DWI',valueList=range(1000),plot=False)
#%%
plt.style.use('default')
MP03.go_cal_ADC()
MP03.imshow_map(plot=plot,crange=[0,2.5],cmap='hot')
# %%
#Ave data
cmap='hot'
crange=[0,2.5]
image = MP03._map[:,:,2]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(2))
fig.canvas.manager.set_window_title('Slice '+ str(2))
multirois = MultiRoi(fig=fig, roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())
Nd=len(roi_names)


#%%
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')
path=os.path.dirname(MP03.path)
img_dir= os.path.join(path,f'{MP03.ID}')
#%%
np.save(img_dir+'_map',image)
np.save(img_dir+'_roi',roi_mask)
#


# %%
