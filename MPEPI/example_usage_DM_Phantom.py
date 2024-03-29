# %% 
#Date Jan 9 2024                    
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
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 500
from t1_fitter import T1_fitter,go_fit_T1
plt.rcParams.update({'axes.titlesize': 'small'})
#%%
plot=False

# %%



#CIRC_ID='CIRC_Phantom_Aug9'
CIRC_ID='CIRC_Phantom_Aug_9th_Diff CIRC_Phantom_Aug_9th_Diff'
#dicomPath=os.path.join(defaultPath,f'20230809_1449_CIRC_Phantom_Aug_9th_Diff_\MP03_DWI\MR000000.dcm')
#dirpath=os.path.dirname(dicomPath)
dicomPath=fr'C:\Research\MRI\MP_EPI\Phantom\CIRC_Phantom_Aug_9th_Diff CIRC_Phantom_Aug_9th_Diff\CIRC_DEVELOPMENT Matthew\MR ep2d_MP03_DTI_T1_T2_DIFF_5slice'
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP03.valueList=[]
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
# %%
#Motion correction
%matplotlib qt
MP03.go_crop()
MP03.go_resize(scale=2)
#MP03.go_moco()
#MP03.imshow_corrected(ID=f'{MP03.ID}_raw',plot=plot)
cropzone=MP03.cropzone
# %%
dicomPath=os.path.join(defaultPath,f'20230809_1449_CIRC_Phantom_Aug_9th_Diff_\MP03_DWI')
dirpath=os.path.dirname(dicomPath)
dicomPath=os.path.join(dirpath,f'MP01_T1_EPI')

print(f'Readding\n{dicomPath}\n')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01_EPI = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327,sigma=100)
MP01_EPI.cropzone=cropzone
MP01_EPI.go_crop()
MP01_EPI.go_resize(scale=2)
fig,axs=MP01_EPI.imshow_corrected(ID='MP01_T1_EPI_raw',plot=plot)

#%%
%matplotlib inline
MP01_EPI.go_ir_fit()
#%%
%matplotlib qt
plt.style.use('default')
MP01_EPI.imshow_map(plot=True)
#%%
plt.style.use('default')
plt.figure(constrained_layout=True)
#MP01Map=MP01_SE._map
MP01Map=MP01_EPI._map[:,:,2]
#MP01Map=np.rot90(MP01Map)
plt.imshow(MP01Map,cmap='magma',vmin=0,vmax=3000, aspect='auto')
plt.axis('off')
img_dir= os.path.join(dirpath,f'MP01_EPI_Slice2')
plt.savefig(img_dir)

#%%
dicomPath=os.path.join(dirpath,f'MP01_T1_SE')

print(f'Readding\n{dicomPath}\n')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01_SE= mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327,sigma=100)
#MP01_SE.go_crop()
MP01_SE.go_resize(scale=2)
fig,axs=MP01_SE.imshow_corrected(ID='MP01_T1_SE_raw',plot=plot)
#%%
MP01_SE.go_ir_fit()

#%%
#finalMap,finalRa,finalRb,finalRes=ir_fit(data=MP01_SE._data,TIlist=MP01_SE.valueList,searchtype='grid',invertPoint=0)
plt.figure(constrained_layout=True)
plt.style.use('default')
#MP01Map=MP01_SE._map

#MP01Map=np.flip(MP01Map,axis=0)
#MP01Map=np.rot90(MP01Map)
plt.imshow(MP01_SE._map.squeeze(),cmap='magma',vmin=0,vmax=3000, aspect='auto')
img_dir= os.path.join(dirpath,f'MP01_SE')
plt.axis('off')
plt.savefig(img_dir)

#%%

dicomPath=os.path.join(dirpath,f'MP02_T2_EPI')
ID = dicomPath.split('\\')[-1]
data,valueList,_=readFolder(dicomPath,sortBy='tval')
MP02_EPI = mapping(data=data,CIRC_ID=CIRC_ID,ID=ID,valueList=valueList)
#MP02_EPI.cropzone=cropzone
MP02_EPI.path=dicomPath
MP02_EPI.valueList=[30,50,80,150]
fig,axs=MP02_EPI.imshow_corrected(ID='MP02_T1_EPI_raw',plot=plot)

# %%
MP02_EPI.cropzone=cropzone
#MP02_EPI.go_crop()
MP02_EPI.go_resize(scale=2)
#MP02.go_moco()
MP02_EPI.imshow_corrected(ID='MP02_EPI_raw',plot=True)
# %%
plt.style.use('default')
MP02_EPI.go_t2_fit()
MP02_EPI.imshow_map(plot=True)
#%%

plt.figure(constrained_layout=True)
#MP01Map=MP01_SE._map
MP02Map=MP02_EPI._map[:,:,2]

#MP01Map=np.rot90(MP01Map)
plt.imshow(MP02Map.squeeze(),cmap='viridis',vmin=0,vmax=120)
plt.axis('off')
img_dir= os.path.join(dirpath,f'MP02_EPI_Slice2')
plt.savefig(img_dir)

#%%
dicomPath=os.path.join(defaultPath,f'20230809_1449_CIRC_Phantom_Aug_9th_Diff_\MP02_T2_SE')
dirpath=os.path.dirname(dicomPath)
dicomPath=os.path.join(dirpath,f'MP02_T2_SE')
ID = dicomPath.split('\\')[-1]
data,valueList,_=readFolder(dicomPath,sortBy='tval')
MP02_SE = mapping(data=data,CIRC_ID=CIRC_ID,ID=ID)
MP02_SE.go_resize(scale=2)
MP02_SE.path=dicomPath
MP02_SE.valueList=[50,80,100,150]
#MP02.go_moco()
MP02_SE.imshow_corrected(ID='MP02_SE_raw',plot=True)

#%%
plt.style.use('default')
MP02_SE.go_t2_fit()


#%%
plt.figure(constrained_layout=True)
plt.style.use('default')
#MP01Map=MP01_SE._map
#MP01Map=np.rot90(MP01Map)
plt.imshow(MP02_SE._map.squeeze(),cmap='viridis',vmin=0,vmax=120)
img_dir= os.path.join(dirpath,f'MP02_SE')
plt.axis('off')
plt.savefig(img_dir)

# %%

MP02_SE._save_nib()
print(MP02_SE.valueList)
#%%
path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'MP02_EPI.mat'))
map_data=map_data['T2']
MP02_EPI.crange=[0,150]
MP02_EPI.cmap='viridis'
MP02_EPI._map= map_data
MP02_EPI.imshow_map()
#%%
map_data=sio.loadmat(os.path.join(path,'MP02_SE.mat'))
map_data=map_data['T2']
plt.imshow(map_data,cmap='viridis',vmin=0,vmax=150)
img_dir= os.path.join(dirpath,f'MP02_SE')
plt.savefig(img_dir)

#%%
#%%__+++add the segmentation
MP01.go_segment_LV(reject=None, image_type="b0")
# save
MP01.save()
# look at stats
MP01.show_calc_stats_LV()
MP01.imshow_overlay()
MP03._update_mask(MP01)
MP03.save()
# look at stats
MP03.show_calc_stats_LV()
MP03.imshow_overlay()

MP02._update_mask(MP01)
MP02.save()
# look at stats
MP02.show_calc_stats_LV()
MP02.imshow_overlay()
# %%
###%%%%%%%plot in mp02
%matplotlib qt 

MP02.go_segment_LV(reject=None, image_type="b0")
# save
MP02.save()
# look at stats
MP02.show_calc_stats_LV()
MP02.imshow_overlay()
MP03._update_mask(MP02)
MP03.save()
# look at stats
MP03.show_calc_stats_LV()
MP03.imshow_overlay()
MP01._update_mask(MP02)
MP01.save()
# look at stats
MP01.show_calc_stats_LV()
MP01.imshow_overlay()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=[0,3])

#%%
data_path=os.path.dirname(dicomPath)
T1path=os.path.join(data_path,r'MR t1map_long_t1_MOCO_T1')
T1_bssfp,_,_  = readFolder(T1path)
T2path=os.path.join(data_path,r'MR t2map_flash_MOCO_T2-2')

T2_bssfp,_,_=readFolder(T2path)

#%%#################################First Run:##################################
#############################Only Need to run one time###################################################
data=T2_bssfp.squeeze()
map_T2=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)
#%%
map_T2.go_resize(scale=2)
map_T2.shape=np.shape(map_T2._data)
map_T2._map=data.squeeze()/10
map_T2.crange=[0,150]
map_T2.cmap='viridis'
map_T2.path=T2path
map_T2.imshow_map()
#%%
data=T1_bssfp.squeeze()
map_T1=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI',CIRC_ID=CIRC_ID)
#%%
map_T1.go_resize(scale=2)
map_T1.path=T1path
map_T1._map=data
map_T2._map=data.squeeze()
crange=[0,3000]
map_T1.crange=crange
map_T1.cmap='magma'
map_T1.shape=np.shape(map._data)
map_T1.imshow_map()
#%%
#################################Second Run:###############################
###########################Please Copy to the Console#####################
#data=T2_bssfp.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)


#data=T1_bssfp_fb.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI_FB',CIRC_ID=CIRC_ID)

#data=T1_bssfp.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI',CIRC_ID=CIRC_ID)
