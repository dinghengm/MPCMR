#%%
#####TO BE IMPLEMENT NOW:
#####ALL T1 TO index 0;
#####Get the transformation from 8000 to 0
#####Register all MP02 MP03 TO 8000 -> Make sure it work before you go
#####APPLY ALL TRANSFORMATION TO ALL MP02 AND MP03 to achieve the thing



# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from MPEPI.libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from MPEPI.libMapping_v12 import readFolder,decompose_LRT,go_ir_fit
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'axes.titlesize': 'small'})
from t1_fitter import T1_fitter,go_fit_T1
#%%
plot=True
# %%
CIRC_ID='CIRC_00405'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737_secondRun\MP03_DWI_DELAY50_Standard')
dirpath=os.path.dirname(dicomPath)
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
# %%
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
#MP03.go_moco()
MP03.imshow_corrected(ID=f'{MP03.ID}_raw',plot=plot)
cropzone=MP03.cropzone
# %%
dicomPath=os.path.join(dirpath,f'MP01_T1')
#CIRC_ID='CIRC_00302'
print(f'Readding\n{dicomPath}\n')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP01 = mapping(data=dicomPath,reject=True,CIRC_ID=CIRC_ID,default=327,sigma=100)
MP01.cropzone=cropzone
MP01.go_crop()
MP01.go_resize(scale=2)
fig,axs=MP01.imshow_corrected(ID='MP01_T1_raw',plot=plot)
#%%

data_8000,_,_=readFolder(dicomPath=os.path.join(dirpath,rf'MR ep2d_MP01_TE_40_bright'))
print(f'Readding\n{dirpath}\n')
#%%
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=True,default=327,sigma=50,sortSlice=False)
print(f'Readding\n{dicomPath}\n')
data0=np.transpose(np.array(data['Slice0']),(1,2,0))
data0=np.concatenate((data0,data_8000[:,:,0]),axis=-1)
data0=np.expand_dims(data0,2)
valueDict['Slice0'].append(8000)
MP01_0 = mapping(data=data0,CIRC_ID=CIRC_ID,ID='MP01_Slice0',valueList=valueDict['Slice0'],datasets=dcmDict['Slice0'])
MP01_0.path=dicomPath
MP01_0.cropzone=cropzone
MP01_0.go_crop()
MP01_0.valueList=valueDict['Slice0']
MP01_0.go_resize(scale=2)
fig,axs=MP01_0.imshow_corrected(plot=plot)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i}\n{valueDict["Slice0"][i]}',fontsize=5)

# %%
#Delete the ones that 
####Be careful for the delete!!!!!!#############################3
MP01_0._delete(d=[0,3,5,6,7,8,10,11])
#%%
#Try momo
def moco(data,obj):
    data_regressed=decompose_LRT(data)
    plt.Figure()
    obj.imshow_corrected(volume=data_regressed,ID=f'{obj.ID}_Regressed',valueList=obj.valueList,plot=True)
    #Show the images
    #Remove the MP01
    Nx,Ny,Nz,_=np.shape(data_regressed)
    data_corrected_temp=np.copy(data_regressed)
    for z in range(Nz):
        data_corrected_temp[:,:,z,:]=obj._coregister_elastix(data_regressed[:,:,z,:],data[:,:,z,:],target_index=-1)
    data_return=data_corrected_temp
    plt.Figure()
    obj.imshow_corrected(volume=data_return,ID=f'{obj.ID}_lrt_moco',valueList=obj.valueList,plot=True)
    return data_return
data_return=moco(MP01_0._data,MP01_0)
#%%
data_tmp0,Transform_0 = MP01_0._coregister_elastix_return_transform(data=np.squeeze(MP01_0._data),target_index=0)
#MP01_0._data=data_tmp2[:,:,np.newaxis,:]
MP01_0.imshow_corrected(volume=data_tmp0[:,:,np.newaxis,:],ID=f'MP01_Slice0_naive_Ind0',plot=plot)
A2=np.copy(data_tmp0)
A2 = data_tmp0/np.max(data_tmp0)*255
img_dir= os.path.join(dirpath,f'{MP01_0.ID}_naive_moco_ind0')
MP01_0.createGIF(img_dir,np.squeeze(A2),fps=5)

#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp0[:,:,np.newaxis,:],TIlist=MP01_0.valueList,searchtype='grid',invertPoint=4)
#MP01_0._map=finalMap
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)
plt.savefig(img_dir)

#%%
MP01_0._delete(d=[5,-2])
data_tmp0_refine,Transform_0_refine = MP01_0._coregister_elastix_return_transform(data=np.squeeze(MP01_0._data),target_index=0)
#MP01_0._data=data_tmp2[:,:,np.newaxis,:]
MP01_0.imshow_corrected(volume=data_tmp0_refine[:,:,np.newaxis,:],ID=f'MP01_Slice0_naive_refine',plot=plot)
A2=np.copy(data_tmp0_refine)
A2 = data_tmp0_refine/np.max(data_tmp0_refine)*255
img_dir= os.path.join(dirpath,f'{MP01_0.ID}_naive_moco_refine')
MP01_0.createGIF(img_dir,np.squeeze(A2),fps=5)

#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp0_refine[:,:,np.newaxis,:],TIlist=MP01_0.valueList,searchtype='grid',invertPoint=4)
#MP01_0._map=finalMap
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)
plt.savefig(img_dir)

#%%
data_tmp0_8000,Transform_0_8000 = MP01_0._coregister_elastix_return_transform(data=np.squeeze(MP01_0._data),target_index=-1)
#MP01_0._data=data_tmp2[:,:,np.newaxis,:]
MP01_0.imshow_corrected(volume=data_tmp0_8000[:,:,np.newaxis,:],ID=f'MP01_Slice0_naive_Ind8000',plot=plot)
A2=np.copy(data_tmp0_8000)
A2 = data_tmp0_8000/np.max(data_tmp0_8000)*255
img_dir= os.path.join(dirpath,f'{MP01_0.ID}_naive_moco_ind8000')
MP01_0.createGIF(img_dir,np.squeeze(A2),fps=5)

#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp0_8000[:,:,np.newaxis,:],TIlist=MP01_0.valueList,searchtype='grid',invertPoint=4)
#MP01_0._map=finalMap
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)
plt.savefig(img_dir)
#%%
MP01_0._data=data_tmp0[:,:,np.newaxis,:]
MP01_0._save_nib()
print(MP01_0.valueList)
MP01_0._map=finalMap
#%%
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=True,default=327,sigma=50,sortSlice=False)
#print(f'Readding\n{dicomPath}\n')
data1=np.transpose(np.array(data['Slice1']),(1,2,0))
data1=np.concatenate((data1,data_8000[:,:,1]),axis=-1)
data1=np.expand_dims(data1,2)
valueDict['Slice1'].append(8000)
MP01_1 = mapping(data=data1,CIRC_ID=CIRC_ID,ID='MP01_Slice1',valueList=valueDict['Slice1'],datasets=dcmDict['Slice1'])
MP01_1.path=dicomPath
MP01_1.cropzone=cropzone
MP01_1.go_crop()
MP01_1.valueList=valueDict['Slice1']
MP01_1.go_resize(scale=2)
fig,axs=MP01_1.imshow_corrected(plot=True)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i}\n{valueDict["Slice1"][i]}',fontsize=5)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_1._delete(d=[0,2,4,6,9,10,11,15])
#%%
data_return=moco(MP01_1._data,MP01_1)
#%%
data_tmp1_8000,Transform_1_8000 = MP01_1._coregister_elastix_return_transform(data=np.squeeze(MP01_1._data),target_index=-1)
#MP01_0._data=data_tmp2[:,:,np.newaxis,:]
MP01_1.imshow_corrected(volume=data_tmp1_8000[:,:,np.newaxis,:],ID=f'MP01_Slice1_naive_Ind8000',plot=plot)
A2=np.copy(data_tmp1_8000)
A2 = data_tmp1_8000/np.max(data_tmp1_8000)*255
img_dir= os.path.join(dirpath,f'{MP01_1.ID}_naive_moco_ind8000')
MP01_0.createGIF(img_dir,np.squeeze(A2),fps=5)

#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp1_8000[:,:,np.newaxis,:],TIlist=MP01_1.valueList,searchtype='grid',invertPoint=4)
#MP01_0._map=finalMap
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)
plt.savefig(img_dir)
#%%
data_tmp1_0,Transform_1_0 = MP01_1._coregister_elastix_return_transform(data=np.squeeze(MP01_1._data),target_index=0)

MP01_1.imshow_corrected(volume=data_tmp1_0[:,:,np.newaxis,:],ID=f'MP01_Slice1_naive_Ind0',plot=True)
A2=np.copy(data_tmp1_0)
A2 = data_tmp1_0/np.max(data_tmp1_0)*255
img_dir= os.path.join(dirpath,f'{MP01_1.ID}_naive_moco_ind0')
MP01_1.createGIF(img_dir,np.squeeze(A2),fps=5)
#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp1_0[:,:,np.newaxis,:],TIlist=MP01_1.valueList,searchtype='grid',invertPoint=4)
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)
plt.savefig(img_dir)


#%%
MP01_1._data=data_tmp1[:,:,np.newaxis,:]
MP01_1._save_nib()
print(MP01_1.valueList)
MP01_1._map=finalMap
#%%
data,valueDict,dcmDict = readFolder(dicomPath=dicomPath,reject=True,default=327,sigma=50,sortSlice=False)
data2=np.transpose(np.array(data['Slice2']),(1,2,0))
data2=np.concatenate((data2,data_8000[:,:,2]),axis=-1)
data2=np.expand_dims(data2,2)
valueDict['Slice2'].append(8000)
MP01_2 = mapping(data=data2,CIRC_ID=CIRC_ID,ID='MP01_Slice2',valueList=valueDict['Slice2'],datasets=dcmDict['Slice2'])
MP01_2.path=dicomPath
MP01_2.cropzone=cropzone
MP01_2.go_crop()
MP01_2.valueList=valueDict['Slice2']
MP01_2.go_resize(scale=2)
fig,axs=MP01_2.imshow_corrected(plot=True)
for i in range(np.shape(axs)[-1]):
    axs[i].set_title(label=f'{i}\n{valueDict["Slice2"][i]}',fontsize=5)

# %%
#Truncation
####Be careful for the delete!!!!!!#############################3
MP01_2._delete(d=[12,11,9,7,5,3,1])
#%%
#LRT
data_return=moco(MP01_2._data,MP01_2)
#%%
data_tmp2_0,Transform_2_0 = MP01_2._coregister_elastix_return_transform(data=np.squeeze(MP01_2._data),target_index=0)
MP01_2.imshow_corrected(volume=data_tmp2_0[:,:,np.newaxis,:],ID=f'MP01_Slice2_naive_Ind0',plot=True)
A2=np.copy(data_tmp2_0)
A2 = data_tmp2_0/np.max(data_tmp2_0)*255
img_dir= os.path.join(dirpath,f'{MP01_2.ID}_naive_moco_Ind0')
MP01_2.createGIF(img_dir,np.squeeze(A2),fps=5)
#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp2_0[:,:,np.newaxis,:]
,TIlist=MP01_2.valueList,searchtype='grid',T1bound=[1,5000],invertPoint=4)
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)

#%%
data_tmp2_8000,Transform_2_8000 = MP01_2._coregister_elastix_return_transform(data=np.squeeze(MP01_2._data),target_index=-1)
MP01_2.imshow_corrected(volume=data_tmp2_8000[:,:,np.newaxis,:],ID=f'MP01_Slice2_naive_Ind8000',plot=True)
A2=np.copy(data_tmp2_8000)
A2 = data_tmp2_8000/np.max(data_tmp2_8000)*255
img_dir= os.path.join(dirpath,f'{MP01_2.ID}_naive_moco_Ind8000')
MP01_2.createGIF(img_dir,np.squeeze(A2),fps=5)
#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=data_tmp2_8000[:,:,np.newaxis,:]
,TIlist=MP01_2.valueList,searchtype='grid',T1bound=[1,5000],invertPoint=4)
plt.figure()
plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)


# %%
#Read MP02
MP01_2._data=data_tmp[:,:,np.newaxis,:]
MP01_2._save_nib()
print(MP01_2.valueList)
MP01_2._map=finalMap
#%%
dicomPath=os.path.join(dirpath,f'MP02_T2')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP02.cropzone=cropzone
# %%
MP02.go_crop()
MP02.go_resize(scale=2)
#MP02.go_moco()
MP02.imshow_corrected(ID='MP02_T2_raw',plot=True)

#%%
##Use the IR70 for all MP02 and MP03 co-registered
shape=np.shape(MP02._data)
data_8000_cor=np.zeros((shape[:3]),dtype=np.float64)
data_8000_cor[:,:,0]=np.squeeze(MP01_0._data[...,0])
data_8000_cor[:,:,1]=np.squeeze(MP01_1._data[...,0])
data_8000_cor[:,:,2]=np.squeeze(MP01_2._data[...,0])
data_8000_tmp=np.copy(np.squeeze(data_8000_cor)[:,:,:,np.newaxis])

#%%
#TEMP:
#Try to implement the LRT
#Use MP01._coregister_elastix()
#Perform on the first images:
#MP02-MP01
#Insert the MP01 to MP02
print('Conducting Moco in MP02')
MP02_temp_Ind0=np.concatenate((data_8000_tmp,MP02._data),axis=-1)
MP02_regressed_Ind0=decompose_LRT(MP02_temp_Ind0)
MP02_temp_list=MP02.valueList.copy()
MP02_temp_list.insert(0,'T1')
MP02.imshow_corrected(volume=MP02_regressed_Ind0,ID='MP02_T2_Combined_Regressed_Ind0',valueList=MP02_temp_list,plot=True)
#Show the images
#Remove the MP01
Nx,Ny,Nz,_=np.shape(MP02_temp_Ind0)
MP02_temp_corrected_temp_Ind0=np.copy(MP02_regressed_Ind0)
for z in range(Nz):
    MP02_temp_corrected_temp_Ind0[:,:,z,:]=MP02._coregister_elastix(MP02_regressed_Ind0[:,:,z,:],MP02_temp_Ind0[:,:,z,:])

MP02.imshow_corrected(ID='MP02_T2_Combined_Ind0',volume=MP02_temp_corrected_temp_Ind0[:,:,:,1::],valueList=MP02_temp_list[1::],plot=True)
MP02._save_nib(data=MP02_temp_corrected_temp_Ind0[:,:,:,1::])
#%%
newshape = (np.shape(MP01_0._data)[0], np.shape(MP01_0._data)[1], data_8000.shape[2], data_8000.shape[3])
data_8000_crop = np.zeros(newshape)
for z in range(np.shape(data_8000)[2]):
    data_8000_crop[:,:,z,:]=imresize(imcrop(data_8000[:,:,z,:], cropzone),(newshape[0],newshape[1]))
data_8000_tmp=np.copy(np.squeeze(data_8000_crop)[:,:,:,np.newaxis])

#%%
#TEMP:
#Try to implement the LRT
#Use MP01._coregister_elastix()
#Perform on the first images:
#MP02-MP01
#Insert the MP01 to MP02
print('Conducting Moco in MP02')
MP02_temp_Ind8000=np.concatenate((data_8000_tmp,MP02._data),axis=-1)
MP02_regressed_Ind8000=decompose_LRT(MP02_temp_Ind8000)
MP02_temp_list=MP02.valueList.copy()
MP02_temp_list.insert(0,'T1')
MP02.imshow_corrected(volume=MP02_regressed_Ind8000,ID='MP02_T2_Combined_Regressed_Ind8000',valueList=MP02_temp_list,plot=True)
#Show the images
#Remove the MP01
Nx,Ny,Nz,_=np.shape(MP02_temp_Ind8000)
MP02_temp_corrected_temp_Ind8000=np.copy(MP02_regressed_Ind8000)
for z in range(Nz):
    MP02_temp_corrected_temp_Ind8000[:,:,z,:]=MP02._coregister_elastix(MP02_regressed_Ind8000[:,:,z,:],MP02_temp_Ind8000[:,:,z,:])

MP02.imshow_corrected(ID='MP02_T2_Combined_Ind8000',volume=MP02_temp_corrected_temp_Ind8000[:,:,:,1::],valueList=MP02_temp_list[1::],plot=True)
MP02._save_nib(data=MP02_temp_corrected_temp_Ind8000[:,:,:,1::])
#%%
MP02._data=MP02_temp_corrected_temp_Ind0[:,:,:,1::]
#%%
#MP03-MP01
#%%
##Use the IR70 for all MP02 and MP03 co-registered
shape=np.shape(MP03._data)
data_8000_cor=np.zeros((shape[:3]),dtype=np.float64)
data_8000_cor[:,:,0]=np.squeeze(MP01_0._data[...,0])
data_8000_cor[:,:,1]=np.squeeze(MP01_1._data[...,0])
data_8000_cor[:,:,2]=np.squeeze(MP01_2._data[...,0])
data_8000_tmp=np.copy(data_8000_cor[:,:,:,np.newaxis])
print('Conducting Moco in MP03')
MP03_temp_Ind0=np.concatenate((data_8000_tmp,MP03._data),axis=-1)
MP03_regressed_Ind0=decompose_LRT(MP03_temp_Ind0)
MP03_temp_list=MP03.valueList.copy()
MP03_temp_list.insert(0,'T1')
MP03.imshow_corrected(volume=MP03_regressed_Ind0,ID='MP03_DWI_Combined_Regressed_Ind0',valueList=MP03_temp_list,plot=True)

MP03_temp_corrected_temp_Ind0=np.copy(MP03_regressed_Ind0)
for z in range(Nz):
    MP03_temp_corrected_temp_Ind0[:,:,z,:]=MP03._coregister_elastix(MP03_regressed_Ind0[:,:,z,:],MP03_temp_Ind0[:,:,z,:])

MP03.imshow_corrected(ID='MP03_Combined_Ind0',volume=MP03_temp_corrected_temp_Ind0[:,:,:,1::],valueList=MP03_temp_list[1::],plot=True)
#%%
newshape = (np.shape(MP01_0._data)[0], np.shape(MP01_0._data)[1], data_8000.shape[2], data_8000.shape[3])
data_8000_crop = np.zeros(newshape)
for z in range(np.shape(data_8000)[2]):
    data_8000_crop[:,:,z,:]=imresize(imcrop(data_8000[:,:,z,:], cropzone),(newshape[0],newshape[1]))
data_8000_tmp=np.copy(np.squeeze(data_8000_crop)[:,:,:,np.newaxis])

#%%
#TEMP:
#Try to implement the LRT
#Use MP01._coregister_elastix()
#Perform on the first images:
#MP02-MP01
#Insert the MP01 to MP02
print('Conducting Moco in MP02')
MP03_temp_Ind8000=np.concatenate((data_8000_tmp,MP03._data),axis=-1)
MP03_regressed_Ind8000=decompose_LRT(MP03_temp_Ind8000)
MP03_temp_list=MP03.valueList.copy()
MP03_temp_list.insert(0,'T1')
MP03.imshow_corrected(volume=MP03_regressed_Ind8000,ID='MP03_Combined_Regressed_Ind8000',valueList=MP03_temp_list,plot=True)
#Show the images
#Remove the MP01
Nx,Ny,Nz,_=np.shape(MP03_temp_Ind8000)
MP03_temp_corrected_temp_Ind8000=np.copy(MP03_regressed_Ind8000)
for z in range(Nz):
    MP03_temp_corrected_temp_Ind8000[:,:,z,:]=MP03._coregister_elastix(MP03_regressed_Ind8000[:,:,z,:],MP03_temp_Ind8000[:,:,z,:])

MP03.imshow_corrected(ID='MP03_Combined_Ind8000',volume=MP03_temp_corrected_temp_Ind8000[:,:,:,1::],valueList=MP03_temp_list[1::],plot=True)
MP03._save_nib(data=MP03_temp_corrected_temp_Ind8000[:,:,:,1::])
#%%
import copy
MP03_Ind0=copy.copy(MP03)
MP03_Ind0._data=MP03_temp_corrected_temp_Ind0[:,:,:,1::]
MP03_Ind8000=copy.copy(MP03)
MP03_Ind8000._data=MP03_temp_corrected_temp_Ind8000[:,:,:,1::]
MP02_Ind0=copy.copy(MP02)
MP02_Ind0._data=MP02_temp_corrected_temp_Ind0[:,:,:,1::]
MP02_Ind8000=copy.copy(MP02)
MP02_Ind8000._data=MP02_temp_corrected_temp_Ind8000[:,:,:,1::]
#%%
#Regress 
#%%
#PLOT MOCO
for obj in [MP02,MP03]:
    Nz=obj.Nz
    A2=np.copy(obj._data)
    for i in range(Nz):
        A2[:,:,i,:] = obj._data[...,i,:]/np.max(obj._data[...,i,:])*255
    A3=np.vstack((A2[:,:,0,:],A2[:,:,1,:],A2[:,:,2,:]))
    img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{obj.ID}_moco_.gif')
    obj.createGIF(img_dir,A3,fps=5)
    obj.save()
for obj in [MP01_0,MP01_1,MP01_2]:
    Nz=obj.Nz
    A2=np.copy(obj._data)
    for i in range(Nz):
        A2[:,:,i,:] = obj._data[...,i,:]/np.max(obj._data[...,i,:])*255
    img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{obj.ID}_moco_.gif')
    obj.createGIF(img_dir,np.squeeze(A2),fps=5)
    obj.save()
#%%
#Get the Maps
ADC=MP03_Ind0.go_cal_ADC()
MP03_Ind0.ID='MP03_DWI_Ind0'
MP03_Ind0._map=ADC*1000
MP03_Ind0.imshow_map()

ADC=MP03_Ind8000.go_cal_ADC()
MP03_Ind8000.ID='MP03_DWI_Ind8000'
MP03_Ind8000._map=ADC*1000
MP03_Ind8000.imshow_map()

#%%
MP02._save_nib()
print(f'MP02:{MP02.valueList}')
#%%
path=os.path.dirname(dicomPath)
map_data=sio.loadmat(os.path.join(path,'mp02_Ind8000.mat'))
map_data=map_data['T2']
MP02_Ind8000._map= map_data
MP02_Ind8000.ID='MP02_DWI_Ind8000'
MP02_Ind8000.imshow_map()


map_data=sio.loadmat(os.path.join(path,'mp02_Ind0.mat'))
map_data=map_data['T2']
MP02_Ind0._map= map_data
MP02_Ind0.ID='MP02_DWI_Ind0'
MP02_Ind0.imshow_map()

#%%

#Load the data
path=os.path.dirname(dicomPath)
newshape = (np.shape(MP01_0._data)[0], np.shape(MP01_0._data)[1], data_8000.shape[2], data_8000.shape[3])
map_data=np.zeros(newshape)
map_data[:,:,0,:]=MP01_0._map
map_data[:,:,1,:]=MP01_1._map
map_data[:,:,2,:]=MP01_2._map
MP01._map= np.squeeze(map_data)
MP01.imshow_map()
map_data=sio.loadmat(os.path.join(path,'MP02_T2.mat'))
map_data=map_data['T2']
plt.figure()
MP02._map= map_data
MP02.imshow_map()
#%%
# %%
%matplotlib qt 

MP02.go_segment_LV(reject=None, image_type="b0")
# save
MP02.save()
# look at stats
MP02.show_calc_stats_LV()
MP02.imshow_overlay()

#%%

MP03._update_mask(MP02)
MP03.save()
# look at stats
MP03.show_calc_stats_LV()
MP03.imshow_overlay()
#%%
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
#Show the T1 T2 Map
data_path=os.path.dirname(dicomPath)
T1_bssfp,_,_  = readFolder(os.path.join(data_path,r'MR t1map_long_t1_3slice_8mm_150_gap_MOCO_T1-2'))

T1_bssfp_fb,_,_=readFolder(os.path.join(data_path,r'MR t1map_long_t1_3slice_8mm_150_gap_free_breathing_MOCO_T1-2'))

T2_bssfp,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_3slice_8mm_150_gap_MOCO_T2'))
T2_bssfp_fb,_,_=readFolder(os.path.join(data_path,r'MR t2map_flash_3slice_8mm_150_gap_free_breathing_MOCO_T2'))

#%%#################################First Run:##################################
#############################Only Need to run one time###################################################
data=T2_bssfp_fb.squeeze()
map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH_FB',CIRC_ID=CIRC_ID)

map.shape=np.shape(map._data)
map.go_crop()
map.go_resize(scale=2)
map.shape=np.shape(map._data)
cropzone=map.cropzone
#%%
#################################Second Run:###############################
###########################Please Copy to the Console#####################
#data=T2_bssfp.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T2_FLASH',CIRC_ID=CIRC_ID)


#data=T1_bssfp_fb.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI_FB',CIRC_ID=CIRC_ID)

#data=T1_bssfp.squeeze()
#map=mapping(data=np.expand_dims(data,axis=-1),ID='T1_MOLLI',CIRC_ID=CIRC_ID)

#%%
map.cropzone=cropzone
map.shape=np.shape(map._data)
map.go_crop()
map.go_resize(scale=2)
map.shape=np.shape(map._data)
#%%
##############################T1#################################
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map._data))
crange=[0,3000]
map.crange=crange
map._map=data_crop.squeeze()
map.cropzone=cropzone
print(map.shape)
map.path=MP01.path
map.go_segment_LV(image_type='map',crange=crange)
# %%
##############################T2#################################
#Truncation and Resize and MOCO
from imgbasics import imcrop
temp = imcrop(data[:,:,0], cropzone)
shape = (temp.shape[0], temp.shape[1], data.shape[2])
data_crop = np.zeros(shape)
for z in tqdm(range(map.Nz)):
    data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
data_crop=imresize(data_crop,np.shape(map._data))
#If T2
#map._map=data_crop
map._map=data_crop.squeeze()/10
crange=[0,150]
map.crange=crange
map.cropzone=cropzone
print(map.shape)
map.path=MP02.path
map.go_segment_LV(image_type='map',crange=crange)
# %%
map.show_calc_stats_LV()
map.imshow_map()

map.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping.csv',crange=crange)

#Save again
#filename=f'{map.path}\Processed\{ID}.mapping'
img_dir= os.path.join(os.path.dirname(dicomPath),f'{map.ID}_p.mapping')
map.save(filename=img_dir)
print('Saved the segmentation sucessfully')
map.imshow_overlay()
# %%
