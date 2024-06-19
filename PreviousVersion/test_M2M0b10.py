


# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline                     
from libMapping_v13 import *  # <--- this is all you need to do diffusion processing
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
defaultPath= r'C:\Research\MRI\MP_EPI'
from t1_fitter import T1_fitter,go_fit_T1
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
#%%
boolenTest=input('Would you like to save you Plot? "Y" and "y" to save')
if boolenTest.lower == 'y':
    plot=True
else:
    plot=False
# %%
#Please try to change to CIRC_ID
CIRC_ID='CIRC_00498'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
dirpath=os.path.dirname(dicomPath)
img_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)
# make directory for saved images if it doesn't exist yet
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
#tmp=MP03.valueList[-3::]*2
#MP03.valueList=MP03.valueList+tmp
#MP03.go_crop()
#cropzone=MP03.cropzone
#MP03.imshow_px()
#%%

#MP03.go_calc_MD()
MP03.go_crop_Auto()
#%%

fig,axs=MP03.imshow_corrected(ID='MP03_raw',plot=plot,path=img_dir,valueList=range(0,1000))
#%%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_DWI')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False)
MP02.go_crop_Auto()
fig,axs=MP02.imshow_corrected(ID='MP02_raw',plot=plot,path=img_dir)
# %%
dicomPath=os.path.join(dirpath,f'MP01_T1')

print(f'Readding\n{dicomPath}\n')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01 = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327,sigma=100)
MP01.go_crop_Auto()








plt.figure(figsize=(9,18), constrained_layout=True)
plt.subplot(3,4,3)
plt.imshow(MP03._data[:,:,0,2],vmin=0,vmax=1500,cmap='gray')
plt.axis('off')
plt.title(f'TE60 b50 M2')
plt.subplot(3,4,4)
plt.imshow(MP03._data[:,:,0,-1],vmin=0,vmax=1500,cmap='gray')
plt.axis('off')
plt.title(f'TE60 b500 M2 SNR{int(SNR_1)}')
plt.subplot(3,4,8)
plt.imshow(MP02._data[:,:,0,8],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE60 b0 SNR{int(SNR_2)}')
plt.axis('off')
plt.subplot(3,4,12)
plt.imshow(MP02._data[:,:,0,9],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE60 b10 M0 SNR{int(SNR_3)}')
plt.axis('off')
plt.subplot(345)
plt.imshow(MP02._data[:,:,0,0],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE30 b0 SNR{int(SNR_0_0)}')
plt.axis('off')
plt.subplot(349)
plt.imshow(MP02._data[:,:,0,1],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE30 b10 M0 SNR{int(SNR_0_1)}')
plt.axis('off')
plt.subplot(346)
plt.imshow(MP02._data[:,:,0,4],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE40 b0')
plt.axis('off')
plt.subplot(3,4,10)
plt.imshow(MP02._data[:,:,0,5],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE40 b10 M0')
plt.axis('off')
plt.subplot(347)
plt.imshow(MP02._data[:,:,0,6],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE50 b0')
plt.axis('off')
plt.subplot(3,4,11)
plt.imshow(MP02._data[:,:,0,7],vmin=0,vmax=1500,cmap='gray')
plt.title(f'TE50 b10 M0')
plt.axis('off')