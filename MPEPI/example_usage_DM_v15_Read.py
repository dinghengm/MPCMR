# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt  
from MPEPI.libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from MPEPI.libMapping_v12 import readFolder,decompose_LRT,go_ir_fit,moco,moco_naive
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
plt.rcParams.update({'axes.titlesize': 'small'})
#%%
plot=False
# %%
CIRC_ID='CIRC_00373'
root_dicom_path=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737')


# image root directory
img_root_dir = os.path.join(root_dicom_path, "__saved_ims")
if not os.path.isdir(img_root_dir):
    os.mkdir(img_root_dir)
# Statas saved 
stats_file = os.path.join(defaultPath, "MPEPI_stats_v2.csv") 

#Read the MP01-MP03
mapList=[]
for dirpath,dirs,files in  os.walk(root_dicom_path):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('.mapping'):
            if path.endswith('p.mapping')==False:
                mapList.append(path)
print(mapList)
#%%

moco_data=sio.loadmat(os.path.join(root_dicom_path,rf'{CIRC_ID}_MOCO'))
#Create the GIF and see the images:

for ind,key in enumerate(moco_data):
    print(key,np.shape(moco_data[key]))
    try:
        data_tmp=moco_data[key]
        img_dir= os.path.join(img_root_dir,f'Slice{ind-3}_Yuchi_moco_.gif')
        A2=np.copy(data_tmp)
        A2=data_tmp/np.max(data_tmp)*255
        mp_tmp=mapping(A2)
        mp_tmp.createGIF(img_dir,A2,fps=5)
    except:
        pass
#%%
MP01_0=mapping(mapList[0])
MP01_1=mapping(mapList[1])
MP01_2=mapping(mapList[2])
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
MP03=mapping(mapList[5])


#%%
MP01_list=[MP01_0,MP01_1,MP01_2]
MPs_list=[MP01,MP02,MP03]
#Get the shape of all data, and then replace the data with corrected
#Read the data
#Renew the dataset:
for obj in MPs_list:
    obj._data=np.copy(obj._raw_data)
for ss,obj_T1 in enumerate(MP01_list):
    key=f'moco_Slice{ss}'
    moco_data_single_slice=moco_data[key]
    Ndtmp=0
    for obj in MPs_list:
        if 'mp01' in obj.ID.lower():
            Ndtmp_end=np.shape(MP01_list[ss]._data)[-1]
            obj_T1._data=moco_data_single_slice[:,:,0:Ndtmp_end]
            print(obj_T1.ID,np.shape(obj_T1._data))
            print('valueList=',obj_T1.valueList)
        else:
            Ndtmp_start=Ndtmp_end
            Ndtmp_end+=np.shape(obj._data)[-1]
            obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
            print(obj.ID,np.shape(obj._data))
            print('valueList=',obj.valueList)

#%%
#Calculate the T1 maps
for ss,obj_T1 in enumerate(MP01_list):
    finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=obj_T1._data[:,:,np.newaxis,:],TIlist=obj_T1.valueList,searchtype='grid',invertPoint=4)
    #MP01_0._map=finalMap
    plt.figure()
    plt.imshow(finalMap,cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_T1')
    plt.savefig(img_dir)
    obj_T1._map=finalMap
    obj_T1.save(filename=os.path.join(root_dicom_path,f'{obj_T1.ID}.mapping'))

#%%
#####Please UPDATE THE METHODS!!!!
MP02._save_nib()
print(MP02.valueList)      
#%%
map_data=sio.loadmat(os.path.join(root_dicom_path,'MP02_T2_Z.mat'))
map_data=map_data['T2']
MP02._map=map_data
for ss in range(MP02.Nz):
    plt.figure()
    plt.imshow(map_data[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_T2')
    plt.savefig(img_dir)
    MP02.save(filename=os.path.join(root_dicom_path,f'{MP02.ID}.mapping'))
#%%
map_data=np.copy(MP02._map)
map_data[:,:,0]=np.squeeze(MP01_0._map)
map_data[:,:,1]=np.squeeze(MP01_1._map)
map_data[:,:,2]=np.squeeze(MP01_2._map)
MP01._map= np.squeeze(map_data)
#%%
Nx,Ny,Nz,Nd=np.shape(MP03._data)
MP03.Nx=Nx
MP03.Ny=Ny
MP03.Nz=Nz
MP03.Nd=Nd
ADC=MP03.go_cal_ADC()
MP03._map=ADC*1000
for ss in range(MP03.Nz):
    plt.figure()
    plt.imshow(MP03._map[:,:,ss],cmap='hot',vmin=0,vmax=3)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_DWI')
    plt.savefig(img_dir)
    MP03.save(filename=os.path.join(root_dicom_path,f'{MP03.ID}.mapping'))

#%%
%matplotlib qt 

MP02.go_segment_LV(reject=None, image_type="b0")
# save
# look at stats

MP02.show_calc_stats_LV()

MP03._update_mask(MP02)
# look at stats
MP03.show_calc_stats_LV()
MP01._update_mask(MP02)
# look at stats
MP01.show_calc_stats_LV()

#%%
%matplotlib inline
# image directory
for obj in MPs_list:

    # view overlay
    num_slice = obj.Nz 
    figsize = (3.4*num_slice, 3)
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    vmin = obj.crange[0]
    vmax = obj.crange[1]
    base_im = obj._data[:, :, :, 0]
    axes=axes.ravel()
    for sl in range(num_slice):
        alpha = obj.mask_lv[..., sl] * 1.0
        axes[sl].set_axis_off()
        axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
        im = axes[sl].imshow(obj._map[..., sl], vmin=vmin, vmax=vmax, cmap=obj.cmap,alpha = 1.0*obj.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes[-1], shrink=1, pad=0.018,aspect=11)
    plt.savefig(os.path.join(img_root_dir, f'{obj.ID}_map'))
    plt.show()  




#%%

MP01.export_stats(filename=stats_file,crange=[0,3000])
MP02.export_stats(filename=stats_file,crange=[0,150])
MP03.export_stats(filename=stats_file,crange=[0,3])


# %%

MP01_0.save(filename=os.path.join(root_dicom_path,f'{MP01_0.ID}.mapping'))
MP01_1.save(filename=os.path.join(root_dicom_path,f'{MP01_1.ID}.mapping'))
MP01_2.save(filename=os.path.join(root_dicom_path,f'{MP01_2.ID}.mapping'))
MP02.save(filename=os.path.join(root_dicom_path,f'{MP02.ID}.mapping'))
MP03.save(filename=os.path.join(root_dicom_path,f'{MP03.ID}.mapping'))

# %%
