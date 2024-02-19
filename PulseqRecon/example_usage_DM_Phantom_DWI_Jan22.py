# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline
import sys
sys.path.append('../MPEPI')
                  
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix

import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI\Phantom'
plt.rcParams['savefig.dpi'] = 500
plt.rcParams.update({'axes.titlesize': 'small'})
#%%
plot=True

# %%
########################Phantom Run1
#CIRC_ID='CIRC_Phantom_Aug9'
CIRC_ID='CIRC_PHANTOM_FEB12 CIRC_PHANTOM_FEB12'
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MR ep2d_MP03_DWI_Z_FINAL1')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M2=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03_M2.imshow_corrected(valueList=range(1000),plot=plot)
#%%
import cv2 as cv

image=np.mean(MP03_M2._data[:,:,1,:],axis=-1)
#img=np.float16(image)
from skimage import measure
gray=cv.normalize(image,None,0,255,cv.NORM_MINMAX)
blurred = cv.GaussianBlur(gray, (3, 3), 0)
blurred=np.uint8(blurred)
edged = cv.Canny(blurred, 40, 200)
#cv.imshow("Original image", image)
plt.imshow(edged)
#%%
kernel = np.ones((1, 1), np.uint8)
img_eroded = cv.erode(edged, kernel, iterations=1)
contours, _ = cv.findContours(img_eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
cv.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
print(len(contours), "objects were found in this image.")
#%%
fig, ax = plt.subplots()
for contour in contours:
    ax.plot(contour[:,0, 1], contour[:,0, 0], linewidth=2)

#%%

MP03_M2.go_cal_ADC()
MP03_M2.imshow_map(plot=plot,crange=[0,2],cmap='hot')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')

#%%
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MR ep2d_MP03_DWI_Z_FINAL2')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M0=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
%matplotlib qt
MP03_M0.imshow_corrected(valueList=range(1000),plot=plot)
#%%
MP03_M0.go_cal_ADC()
MP03_M0.imshow_map(plot=plot,crange=[0,2],cmap='hot')

#%%
##########################Phantom2####################################

#CIRC_ID='CIRC_Phantom_Aug9'
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MR ep2d_MP03_DWI_Z_FINAL3')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M2=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03_M2.imshow_corrected(valueList=range(1000),plot=plot)

#%%

MP03_M2.go_cal_ADC()
MP03_M2.imshow_map(plot=plot,crange=[0,2],cmap='hot')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')


#%%
##########################Phantom3####################################

#CIRC_ID='CIRC_Phantom_Aug9'
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MR ep2d_MP03_DWI_Z_FINAL1_1ave')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M2=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03_M2.imshow_corrected(valueList=range(1000),plot=plot)

#%%

MP03_M2.go_cal_ADC()
MP03_M2.imshow_map(plot=plot,crange=[0,2],cmap='hot')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')

#%%
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MP03_M0_3')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M0=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03_M0.imshow_corrected(valueList=range(1000),plot=plot)
#%%
MP03_M0.go_cal_ADC()
MP03_M0.imshow_map(plot=plot)

#%%
#################15 SE#############################3
import copy
MP03_SE_data=np.load('C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\DWI_6_Jan22_b50_15.npy')
MP03_SE_tmp=np.rot90(MP03_SE_data)
MP03_SE_tmp=np.flip(MP03_SE_tmp,axis=1)
MP03_SE=copy.copy(MP03_M2)
MP03_SE._update_data(MP03_SE_tmp)
MP03_SE.ID=str(MP03_M2.ID+'SE_15')
MP03_SE.bval=np.array([50,50,50,500,500,500])
MP03_SE.bvec=MP03_M2.bvec[0:6]
MP03_SE.imshow_corrected(valueList=range(1000),plot=plot)
plt.show()

#%%
MP03_SE.go_cal_ADC()
crange=MP03_SE.crange
cmap=MP03_SE.cmap
plt.imshow(MP03_SE._map.squeeze(),vmin=0,vmax=3,cmap=cmap)
img_dir= os.path.join(os.path.dirname(MP03_SE.path),f'{MP03_SE.ID}')
plt.savefig(img_dir)
#%%
#################2 SE#############################
import copy
MP03_SE_data=np.load('C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\DWI_6_Jan22_b50_2.npy')
MP03_SE_tmp=np.rot90(MP03_SE_data)
MP03_SE_tmp=np.flip(MP03_SE_tmp,axis=1)
MP03_SE=copy.copy(MP03_M2)
MP03_SE._update_data(MP03_SE_tmp)
MP03_SE.ID=str(MP03_M2.ID+'SE_2')
MP03_SE.bval=np.array([50,50,50,500,500,500])
MP03_SE.bvec=MP03_M2.bvec[0:6]
MP03_SE.imshow_corrected(valueList=range(1000),plot=plot)
plt.show()

#%%
MP03_SE.go_cal_ADC()
crange=MP03_SE.crange
cmap=MP03_SE.cmap
plt.imshow(MP03_SE._map.squeeze(),vmin=0,vmax=3,cmap=cmap)
img_dir= os.path.join(os.path.dirname(MP03_SE.path),f'{MP03_SE.ID}')
plt.savefig(img_dir)
#%%
#################3 SE#############################
import copy
MP03_SE_data=np.load('C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\DWI_6_Jan22_b50_1.npy')
MP03_SE_tmp=np.rot90(MP03_SE_data)
MP03_SE_tmp=np.flip(MP03_SE_tmp,axis=1)
MP03_SE=copy.copy(MP03_M2)
MP03_SE._update_data(MP03_SE_tmp)
MP03_SE.ID=str(MP03_M2.ID+'SE_1')
MP03_SE.bval=np.array([50,50,50,500,500,500])
MP03_SE.bvec=MP03_M2.bvec[0:6]
MP03_SE.imshow_corrected(valueList=range(1000),plot=plot)
plt.show()

#%%
MP03_SE.go_cal_ADC()
crange=MP03_SE.crange
cmap=MP03_SE.cmap
plt.imshow(MP03_SE._map.squeeze(),vmin=0,vmax=3,cmap=cmap)
img_dir= os.path.join(os.path.dirname(MP03_SE.path),f'{MP03_SE.ID}')
plt.savefig(img_dir)
#%%
#################3 SE#############################3
import copy
MP03_SE_data=np.load('C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\DWI_6_Jan22_b50_3.npy')
MP03_SE_tmp=np.rot90(MP03_SE_data)
MP03_SE_tmp=np.flip(MP03_SE_tmp,axis=1)
MP03_SE=copy.copy(MP03_M2)
MP03_SE._update_data(MP03_SE_tmp)
MP03_SE.ID=str(MP03_M2.ID+'SE_3')
MP03_SE.bval=np.array([50,50,50,500,500,500])
MP03_SE.bvec=MP03_M2.bvec[0:6]
MP03_SE.imshow_corrected(valueList=range(1000),plot=plot)
plt.show()

#%%
MP03_SE.go_cal_ADC()
crange=MP03_SE.crange
cmap=MP03_SE.cmap
plt.imshow(MP03_SE._map.squeeze(),vmin=0,vmax=3,cmap=cmap)
img_dir= os.path.join(os.path.dirname(MP03_SE.path),f'{MP03_SE.ID}')
plt.savefig(img_dir)
# %%
#####Finally we use the phantom 3 for comparason##########################3
MP03_M0.save()
MP03_M2.save()
MP03_SE.save()
#%%

#CIRC_ID='CIRC_Phantom_Aug9'
CIRC_ID='CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final'
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MP03_M2_3')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M2=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)



#%%
dicomPath=os.path.join(defaultPath,CIRC_ID,f'MP03_M0_3')
#dirpath=os.path.dirname(dicomPath)
#dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\MP03_M2_4Run'
MP03_M0=mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)

# %%
MP03_SE_data=np.load('C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final\DWI_6_Jan22_b50_3.npy')
MP03_SE_tmp=np.rot90(MP03_SE_data)
MP03_SE_tmp=np.flip(MP03_SE_tmp,axis=1)
MP03_SE=copy.copy(MP03_M2)
MP03_SE._update_data(MP03_SE_tmp[:,:,np.newaxis,:])
MP03_SE.ID=str(MP03_M2.ID+'SE_3')
MP03_SE.bval=np.array([50,50,50,500,500,500])
MP03_SE.bvec=MP03_M2.bvec[0:6]
MP03_SE.Nz=1
#MP03_SE.imshow_corrected(valueList=range(1000),plot=plot)

#%%
#Resize
MP03_M0.go_crop()
MP03_M0.go_resize(scale=2)
cropzone=MP03_M0.cropzone
MP03_M2.cropzone=cropzone
MP03_M2.go_crop()
MP03_M2.go_resize(scale=2)
#%%
#%%

MP03_M2.go_cal_ADC()
MP03_M2.imshow_map(plot=plot,ID=f'map_{MP03_M2.ID}_Cropped')
#%%
MP03_M0.go_cal_ADC()
MP03_M0.imshow_map(plot=plot,ID=f'map_{MP03_M0.ID}_Cropped')

#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
#%%
MP03_SE.cropzone=(0, 48, 66, 71)
MP03_SE.go_crop()
MP03_SE.go_resize(scale=2)
#%%
MP03_SE.go_cal_ADC()
crange=MP03_SE.crange
cmap=MP03_SE.cmap
plt.imshow(MP03_SE._map.squeeze(),vmin=0,vmax=3,cmap=cmap)
img_dir= os.path.join(os.path.dirname(MP03_SE.path),f'map_{MP03_SE.ID}_Cropped')
plt.savefig(img_dir)
#%%
MP03_M0.save()
MP03_M2.save()
MP03_SE.save()
#%%


MP03_M0._save_nib(data=MP03_M0._map,ID=f'map_{MP03_M0.ID}.nii.gz')
MP03_M2._save_nib(data=MP03_M2._map,ID=f'map_{MP03_M2.ID}.nii.gz')
MP03_SE._save_nib(data=MP03_SE._map,ID=f'map_{MP03_SE.ID}.nii.gz')






# %%
#####Could not use below####################################

MP03_M0.go_segment_LV(roi_names=['0', '1','2','3','4','5'],crange=[0,3],cmap='hot',z=1)
# %%
self.CoM=np.copy(segmented.CoM)
self.mask_lv=np.copy(segmented.mask_lv)
self.mask_endo=np.copy(segmented.mask_endo)
self.mask_epi=np.copy(segmented.mask_epi)
try:
    self.mask_septal=np.copy(segmented.mask_septal)
    self.mask_lateral=np.copy(segmented.mask_lateral)
except:
    print('Not implement septal and lateral')

#%%
MP03_SE.go_segment_LV(roi_names=['0', '1','2','3','4','5'],crange=[0,3],cmap='hot')

#%%
MP03_M0.show_calc_stats_LV()
MP03_M2.show_calc_stats_LV()
MP03_SE.show_calc_stats_LV()
#MP01_post.show_calc_stats_LV()
#%%
MP03_M0.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Phantom.csv',crange=[0,3000])
MP03_M2.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Phantom.csv',crange=[0,150])
MP03_SE.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Phantom.csv',crange=[0,3])

# %%
