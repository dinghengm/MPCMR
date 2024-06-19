#%%
%matplotlib qt                      
from MPEPI.libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from MPEPI.libMapping_v12 import readFolder,decompose_LRT,go_ir_fit,moco,moco_naive,bmode
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
plt.rcParams.update({'axes.titlesize': 'small'})
#%%
plot=True
#%%
CIRC_ID='CIRC_00418'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI_Z_DELAY100')
dirpath=os.path.dirname(dicomPath)
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#cropzone=MP03.cropzone

# %%
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
#MP03.go_moco()
#MP03.imshow_corrected(ID=f'{MP03.ID}_raw',plot=plot)
cropzone=MP03.cropzone
#%%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2_Z_Delay100')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP02.cropzone=cropzone
# %%
MP02.go_crop()
MP02.go_resize(scale=2)
#MP02.go_moco()
MP02.imshow_corrected(ID='MP02_T2_raw',plot=True)
#%%
data_8000,_,_=readFolder(dicomPath=os.path.join(dirpath,rf'MR ep2d_MP01_TE_40_bright_Delay100'))
print(f'Readding\n{dirpath}\n')
#%%
newshape = (np.shape(MP02._data)[0], np.shape(MP02._data)[1], data_8000.shape[2], data_8000.shape[3])
data_8000_crop = np.zeros(newshape)
for z in range(np.shape(data_8000)[2]):
    data_8000_crop[:,:,z,:]=imresize(imcrop(data_8000[:,:,z,:], cropzone),(newshape[0],newshape[1]))
data_8000_tmp=np.copy(np.squeeze(data_8000_crop)[:,:,:,np.newaxis])
# %%
print('Conducting Moco in MP02')

MP02_temp_Ind0=np.concatenate((data_8000_tmp,MP02._data),axis=-1)
MP02_temp_list=MP02.valueList.copy()
MP02_temp_list.insert(0,'40')
data_return02_lrt=moco(MP02_temp_Ind0,MP02,MP02_temp_list)

#%%
##Use the naive for registeration
data_return02_naive=moco_naive(MP02_temp_Ind0,MP02,MP02_temp_list)


#%%
MP02._save_nib(data=data_return02_lrt[:,:,:,1::],ID='MP02_T2_Z_Delay100_Naive')
print(MP02.valueList)
#%%


#MP03-MP01
print('Conducting Moco in MP03')
MP03_temp_Ind0=np.concatenate((data_8000_tmp,MP03._data),axis=-1)

MP03_temp_list=MP03.valueList.copy()
MP03_temp_list=[i for i in range(np.shape(MP03._data)[-1])]
MP03_temp_list.insert(0,'TE')
data_return03_lrt=moco(MP03_temp_Ind0,MP03,MP03_temp_list)
#%%
##Use the naive for registeration
data_return03_naive=moco_naive(MP03_temp_Ind0,MP03,MP03_temp_list)


#%%
MP02._data=data_return02_lrt[:,:,:,1::]
MP03._data=data_return03_lrt[:,:,:,1::]

#%%
def go_cal_ADC(obj):
        #Assume the first 3 is b50, the later 3 is b500
        print("Starting calculation of ADC")
        ADC_temp=np.zeros((obj.Nx,obj.Ny,obj.Nz,3))
        S50=np.zeros((obj.Nx,obj.Ny))
        S500=np.zeros((obj.Nx,obj.Ny))
        for z in range(obj.Nz):
            for d in range(3):
                S50= obj._data[...,z,d]
                S500= obj._data[...,z,d+6]
                ADC_temp[:,:,z,d]=-1/450 * np.log(S500/S50)
        #S50=np.cbrt(S50)
        #S500=self._data[...,-1]*self._data[...,-2]*self._data[...,-3]
        #S500=np.cbrt(S500)
        ADCmap=np.zeros((obj.Nx,obj.Ny,obj.Nz))
        ADCmap=np.mean(ADC_temp,axis=-1)
        obj.ADC=ADCmap
        return ADCmap

#%%


#Get the Maps
ADC=go_cal_ADC(MP03)
MP03._map=ADC*1000
MP03.imshow_map()

#%%
map_data=sio.loadmat(os.path.join(dirpath,'MP02_T2_Z_Naive_moco.mat'))
map_data=map_data['T2']
plt.figure()
MP02._map= map_data
MP02.imshow_map()


# %%
MP02.save()
MP03.save()
# %%
