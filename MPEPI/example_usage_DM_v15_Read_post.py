# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import readFolder,decompose_LRT,moco,moco_naive
import numpy as np
import matplotlib.pyplot as plt
import os
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

#%%
plot=True
# %%
CIRC_ID='CIRC_00438'
img_root_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)

# image root directory
# Statas saved 
stats_file = os.path.join(defaultPath, "MPEPI_stats_v2.csv") 

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
#See the moco methods

img_moco_dir='C:\Research\MRI\MP_EPI\Moco_Dec6\MOCO'
moco_data=h5py.File(os.path.join(img_moco_dir,rf'{CIRC_ID}_MOCO.mat'),'r')
#Create the GIF and see the images:

for ind,key in enumerate(moco_data):
    print(key,np.shape(moco_data[key]))
    try:
        data_tmp=np.transpose(moco_data[key],(2,1,0))
        img_path= os.path.join(img_root_dir,f'Slice{ind-3}_Yuchi_moco_.gif')
        A2=np.copy(data_tmp)
        A2=data_tmp/np.max(data_tmp)*255
        mp_tmp=mapping(A2)
        mp_tmp.createGIF(img_path,A2,fps=5)
        mp_tmp.imshow_corrected(volume=data_tmp[:,:,np.newaxis,:],valueList=range(1000),ID=f'Slice{ind-3}_Yuchi_moco',plot=plot,path=img_root_dir)

    except:

        pass
#%%

MP01=mapping(mapList[0])
MP01_post=mapping(mapList[1])
MP02=mapping(mapList[2])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[3])
MPs_list=[MP01,MP02,MP03,MP01_post]
#%%
#The MP01 is at the end of the dimension
                             
for obj in MPs_list:
    print(obj.shape)
    obj._data=np.copy(obj._raw_data)
Nz=obj.Nz

#Get the shape of all data, and then replace the data with corrected
#Read the data
#Renew the dataset:
for ss in range(Nz):
    Ndtmp_end=0
    key=f'moco_Slice{ss}'
    moco_data_single_slice=np.transpose(moco_data[key],(2,1,0))
    for obj in MPs_list:
        Ndtmp_start=Ndtmp_end
        Ndtmp_end+=np.shape(obj._data)[-1]
        obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
        print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
        #print('valueList=',obj.valueList)

for obj in MPs_list:
    if 'mp03' in obj.ID.lower():
            obj.imshow_corrected(ID=f'{obj.ID}_moco',plot=plot,path=img_root_dir,valueList=obj.bval)
    else:

        obj.imshow_corrected(ID=f'{obj.ID}_moco',plot=plot,path=img_root_dir)
    obj.go_create_GIF(path_dir=str(img_root_dir))
    obj.go_crop_Auto()
    obj.go_resize()
#%%
#Save the file as M
stats_file = os.path.join(os.path.dirname(img_root_dir), "MPEPI_stats_v2.csv") 

for obj in MPs_list:
    obj.save(filename=os.path.join(img_root_dir,f'{obj.ID}_m.mapping'))
    keys=['CIRC_ID','ID','valueList']
    if 'mp02' in obj.ID.lower():
        stats=[obj.CIRC_ID,obj.ID,str(obj.valueList)]
    elif 'mp03' in obj.ID.lower():
        stats=[obj.CIRC_ID,obj.ID,str(obj.bval)]
    else:
        continue
    data=dict(zip(keys,stats))
    cvsdata=pd.DataFrame(data, index=[0])
    if os.path.isfile(stats_file):    
        cvsdata.to_csv(stats_file, index=False, header=False, mode='a')
    else:
        cvsdata.to_csv(stats_file, index=False)


#%%

#---...-----
#Please calculate the maps in a loop
finalMap,finalRa,finalRb,finalRes=MP01.go_ir_fit(searchtype='grid',invertPoint=4)
MP01._map=finalMap
MP01.imshow_map(path=img_root_dir,plot=plot)
MP01.save(filename=os.path.join(img_root_dir,f'{MP01.ID}_p.mapping'))
finalMap,finalRa,finalRb,finalRes=MP01_post.go_ir_fit(searchtype='grid',invertPoint=4)
MP01_post._map=finalMap
MP01_post.imshow_map(path=img_root_dir,plot=plot)
MP01_post.save(filename=os.path.join(img_root_dir,f'{MP01_post.ID}_p.mapping'))

#%%
MP03.go_cal_ADC()
MP03.imshow_map(path=img_root_dir,plot=plot)
MP03.save(filename=os.path.join(img_root_dir,f'{MP03.ID}_p.mapping'))

#%%
MP02.go_t2_fit()
MP02.imshow_map(path=img_root_dir,plot=plot)
MP02.save(filename=os.path.join(img_root_dir,f'{MP02.ID}_p.mapping'))


# %%
