# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
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
#%%
plot=True
# %%
CIRC_ID='CIRC_00373'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI_Z')
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
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2_Z')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False)
fig,axs=MP02.imshow_corrected(ID='MP02_raw',plot=plot,path=img_dir)
# %%
dicomPath=os.path.join(dirpath,f'MP01_T1')

print(f'Readding\n{dicomPath}\n')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01 = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327,sigma=100)
#MP01.go_crop_Auto()

#MP01.imshow_px()
fig,axs=MP01.imshow_corrected(ID='MP01_T1_raw',plot=plot,path=img_dir)
#%%
#%%

data_8000,_,_=readFolder(dicomPath=os.path.join(dirpath,rf'MR ep2d_MP01_TE_40_bright_Z'))
print(f'Readding\n{dirpath}\n')
#%%
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=False,default=270,sigma=75,sortSlice=False)
data0=np.transpose(np.array(data['Slice0']),(1,2,0))
data0=np.concatenate((data0,data_8000[:,:,0]),axis=-1)
data0=np.expand_dims(data0,2)
valueDict['Slice0'].append(8000)
MP01_0 = mapping(data=data0,CIRC_ID=CIRC_ID,ID='MP01_Slice0',valueList=valueDict['Slice0'],datasets=dcmDict['Slice0'])
MP01_0.path=dicomPath
MP01_0.go_crop_Auto()
MP01_0.go_resize(scale=2)
#%%
Half=int(MP01_0.Nd/2)
fig,axs=MP01_0.imshow_corrected(volume=MP01_0._data[:,:,:,0:Half],plot=False)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i}\n{valueDict["Slice0"][i]}')

fig,axs=MP01_0.imshow_corrected(volume=MP01_0._data[:,:,:,Half::],plot=False)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i+Half}\n{valueDict["Slice0"][i+Half]}')

# %%
#Delete the ones that 
####Be careful for the delete!!!!!!#############################3
MP01_0._delete(d=[0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,22,23,25,27,28,29,31,32,33,35,36,37,38,40])
fig,axs=MP01_0.imshow_corrected(ID='MP01_0_T1_seletive1',plot=plot,path=img_dir)
#%%
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=False,default=270,sigma=75,sortSlice=False)
data1=np.transpose(np.array(data['Slice1']),(1,2,0))
data1=np.concatenate((data1,data_8000[:,:,1]),axis=-1)
data1=np.expand_dims(data1,2)
valueDict['Slice1'].append(8000)
MP01_1 = mapping(data=data1,CIRC_ID=CIRC_ID,ID='MP01_Slice1',valueList=valueDict['Slice1'],datasets=dcmDict['Slice1'])
MP01_1.path=dicomPath
MP01_1.go_crop_Auto()
MP01_1.go_resize(scale=2)

#%%
fig,axs=MP01_1.imshow_corrected(volume=MP01_1._data[:,:,:,0:Half],plot=False)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i}\n{valueDict["Slice1"][i]}')

fig,axs=MP01_1.imshow_corrected(volume=MP01_1._data[:,:,:,Half::],plot=False)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i+Half}\n{valueDict["Slice1"][i+Half]}')
# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_1._delete(d=[0,2,3,5,6,7,9,12,13,15,16,18,20,21,23,25,27,29,30,31,32,33,34,35,36,37,38,39])
fig,axs=MP01_1.imshow_corrected(ID='MP01_1_T1_seletive1',plot=plot,path=img_dir)

#%%

data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=False,default=270,sigma=75,sortSlice=False)
data2=np.transpose(np.array(data['Slice2']),(1,2,0))
data2=np.concatenate((data2,data_8000[:,:,2]),axis=-1)
data2=np.expand_dims(data2,2)
valueDict['Slice2'].append(8000)
MP01_2 = mapping(data=data2,CIRC_ID=CIRC_ID,ID='MP01_Slice2',valueList=valueDict['Slice2'],datasets=dcmDict['Slice2'])
MP01_2.path=dicomPath
MP01_2.go_crop_Auto()
MP01_2.go_resize(scale=2)
#%%
fig,axs=MP01_2.imshow_corrected(volume=MP01_2._data[:,:,:,0:Half],plot=False)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i}\n{valueDict["Slice2"][i]}')

fig,axs=MP01_2.imshow_corrected(volume=MP01_2._data[:,:,:,Half::],plot=False)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i+Half}\n{valueDict["Slice2"][i+Half]}')
# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_2._delete(d=[1,2,3,6,7,8,10,11,14,15,16,17,18,20,22,23,25,27,28,29,30,31,32,33,35,36,37,38,39,40
])
fig,axs=MP01_2.imshow_corrected(ID='MP01_2_T1_seletive1',plot=plot,path=img_dir)
#%%
#SAVE MP01MP02MP3
from numpy.core.records import fromarrays
from scipy.io import savemat

mdict={}
mdict['MP01_Slice0']=MP01_0._raw_data
mdict['MP01_Slice1']=MP01_1._raw_data
mdict['MP01_Slice2']=MP01_2._raw_data
mdict['MP02_Slice0']=MP02._raw_data[:,:,0,:][:,:,np.newaxis,:]
mdict['MP02_Slice1']=MP02._raw_data[:,:,1,:][:,:,np.newaxis,:]
mdict['MP02_Slice2']=MP02._raw_data[:,:,2,:][:,:,np.newaxis,:]
mdict['MP03_Slice0']=MP03._raw_data[:,:,0,:][:,:,np.newaxis,:]
mdict['MP03_Slice1']=MP03._raw_data[:,:,1,:][:,:,np.newaxis,:]
mdict['MP03_Slice2']=MP03._raw_data[:,:,2,:][:,:,np.newaxis,:]
savepath=rf'{img_dir}\{MP03.CIRC_ID}.mat'
savemat(savepath,mdict)
##Can change if something wrong
savemat(rf'C:\Research\MRI\MP_EPI\Moco_Dec6\{MP03.CIRC_ID}.mat',mdict)
MP01.save(filename=os.path.join(img_dir,f'{MP01.ID}.mapping'))
MP01_0.save(filename=os.path.join(img_dir,f'{MP01_0.ID}.mapping'))
MP01_1.save(filename=os.path.join(img_dir,f'{MP01_1.ID}.mapping'))
MP01_2.save(filename=os.path.join(img_dir,f'{MP01_2.ID}.mapping'))
MP02.save(filename=os.path.join(img_dir,f'{MP02.ID}.mapping'))
MP03.save(filename=os.path.join(img_dir,f'{MP03.ID}.mapping'))
#%%
##################PIG data:###############################
#SAVE MP01MP02MP3
from numpy.core.records import fromarrays
from scipy.io import savemat
mdict={}

mdict['MP01_Slice0']=MP01._raw_data[:,:,0,:][:,:,np.newaxis,:]
mdict['MP01_Slice1']=MP01._raw_data[:,:,1,:][:,:,np.newaxis,:]
mdict['MP01_Slice2']=MP01._raw_data[:,:,2,:][:,:,np.newaxis,:]
mdict['MP02_Slice0']=MP02._raw_data[:,:,0,:][:,:,np.newaxis,:]
mdict['MP02_Slice1']=MP02._raw_data[:,:,1,:][:,:,np.newaxis,:]
mdict['MP02_Slice2']=MP02._raw_data[:,:,2,:][:,:,np.newaxis,:]
mdict['MP03_Slice0']=MP03._raw_data[:,:,0,:][:,:,np.newaxis,:]
mdict['MP03_Slice1']=MP03._raw_data[:,:,1,:][:,:,np.newaxis,:]
mdict['MP03_Slice2']=MP03._raw_data[:,:,2,:][:,:,np.newaxis,:]
MP01.save(filename=os.path.join(img_dir,f'{MP01.ID}.mapping'))
MP02.save(filename=os.path.join(img_dir,f'{MP02.ID}.mapping'))
MP03.save(filename=os.path.join(img_dir,f'{MP03.ID}.mapping'))

try:
    dicomPath=os.path.join(dirpath,f'MP01_T1_post')

    print(f'Readding\n{dicomPath}\n')
    ID = os.path.dirname(dicomPath).split('\\')[-1]
    MP01_post = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,default=327,sigma=100)
    #MP01.go_crop_Auto()

    #MP01.imshow_px()
    fig,axs=MP01_post.imshow_corrected(ID='MP01_T1_raw',plot=plot,path=img_dir)
    mdict['MP01_post_Slice0']=MP01_post._raw_data[:,:,0,:][:,:,np.newaxis,:]
    mdict['MP01_post_Slice1']=MP01_post._raw_data[:,:,1,:][:,:,np.newaxis,:]
    mdict['MP01_post_Slice2']=MP01_post._raw_data[:,:,2,:][:,:,np.newaxis,:]

    MP01_post.save(filename=os.path.join(img_dir,f'{MP01_post.ID}.mapping'))
except:
    pass


savepath=rf'{img_dir}\{MP03.CIRC_ID}.mat'
savemat(savepath,mdict)
##Can change if something wrong
savemat(rf'C:\Research\MRI\MP_EPI\Moco_Dec6\{MP03.CIRC_ID}.mat',mdict)

#%%
print(savepath)
#%%
from scipy.io import loadmat
a=loadmat(savepath)
#%%
print(MP02.CIRC_ID)

for items in a:
    print(items,f'{np.shape(a[items])}')

# %%