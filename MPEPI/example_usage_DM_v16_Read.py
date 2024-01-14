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
CIRC_ID='CIRC_00446'
img_root_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Jan_12_2024_moco",CIRC_ID)
if not os.path.exists(saved_img_root_dir):
            os.mkdir(saved_img_root_dir)

# image root directory
# Statas saved 
stats_file = os.path.join(defaultPath, "MPEPI_stats_v16.csv") 

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
        mp_tmp.createGIF(img_path,A2,fps=5)
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
            obj_T1.go_crop_Auto()
            obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped',plot=plot,path=saved_img_root_dir)
            obj_T1.go_create_GIF(path_dir=str(saved_img_root_dir))
            obj_T1._update_data(moco_data_single_slice[:,:,0:Ndtmp_end])
            print(obj_T1.ID,np.shape(obj_T1._data))
            print('valueList=',obj_T1.valueList)
        else:
            Ndtmp_start=Ndtmp_end
            Ndtmp_end+=np.shape(obj._data)[-1]
            obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
            print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
            #print('valueList=',obj.valueList)
'''
for obj in MP01_list:
    obj.go_crop_Auto()
    obj.go_resize()
for obj in MPs_list:
    obj.go_crop_Auto()
    obj.go_resize()
'''
#%%
#Replace the between frames with the original frames
#Conservative in 800-900 only
for ss,obj_T1 in enumerate(MP01_list):
    valueArray=np.array(obj_T1.valueList)
    arrayInd=np.where(np.logical_and(valueArray>=700,valueArray<=1200))
    
    obj_T1._data[...,arrayInd]=obj_T1._raw_data[...,arrayInd]
    obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_updated',plot=plot,path=saved_img_root_dir)


#%%
#Save the file as M
stats_file = os.path.join(os.path.dirname(saved_img_root_dir), "MPEPI_stats_v2.csv") 

for obj in MP01_list:
    obj.save(filename=os.path.join(saved_img_root_dir,f'{obj.ID}_m.mapping'))
    keys=['CIRC_ID','ID','valueList','shape']
    stats=[obj.CIRC_ID,obj.ID,str(obj.valueList),str(np.shape(obj._data))]
    data=dict(zip(keys,stats))
    cvsdata=pd.DataFrame(data, index=[0])
    if os.path.isfile(stats_file):    
        cvsdata.to_csv(stats_file, index=False, header=False, mode='a')
    else:
        cvsdata.to_csv(stats_file, index=False)

for obj in MPs_list:
    obj.save(filename=os.path.join(saved_img_root_dir,f'{obj.ID}_m.mapping'))
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
#%%
#Calculate the T1 maps
for ss,obj_T1 in enumerate(MP01_list):
    finalMap,finalRa,finalRb,finalRes=obj_T1.go_ir_fit(searchtype='grid',invertPoint=4)
    plt.figure()
    plt.axis('off')
    plt.imshow(finalMap.squeeze(),cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_T1')
    plt.savefig(img_dir)
    MP01_0._map=finalMap
    obj_T1.save(filename=os.path.join(img_root_dir,f'{obj_T1.ID}_p.mapping'))

#%%
MP02._update()
MP02.go_create_GIF(path_dir=str(img_root_dir))
MP02.imshow_corrected(ID=f'MP02_Corrected',plot=plot,path=img_root_dir)

#%%
MP03._update()
MP03.go_create_GIF(path_dir=str(img_root_dir))
MP03.imshow_corrected(ID=f'MP03_Corrected',valueList=MP03.bval,plot=plot,path=img_root_dir)
MP03.go_cal_ADC()
#%%
MP02.go_t2_fit()


#%%
for ss in range(MP02.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP02._map[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_T2')
    
    plt.savefig(img_dir)
MP02.save(filename=os.path.join(img_root_dir,f'{MP02.ID}_p.mapping'))
#%%
map_data=np.copy(MP02._map)
map_data[:,:,0]=np.squeeze(MP01_0._map)
map_data[:,:,1]=np.squeeze(MP01_1._map)
map_data[:,:,2]=np.squeeze(MP01_2._map)
MP01._map= np.squeeze(map_data)
#%%
Nx,Ny,Nz,Nd=np.shape(MP03._data)
for ss in range(MP03.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP03.ADC[:,:,ss],cmap='hot',vmin=0,vmax=3)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_DWI')
    plt.savefig(img_dir)
MP03.save(filename=os.path.join(img_root_dir,f'{MP03.ID}_p.mapping'))

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

MP01_0.save(filename=os.path.join(img_root_dir,f'{MP01_0.ID}_p.mapping'))
MP01_1.save(filename=os.path.join(img_root_dir,f'{MP01_1.ID}_p.mapping'))
MP01_2.save(filename=os.path.join(img_root_dir,f'{MP01_2.ID}_p.mapping'))
MP02.save(filename=os.path.join(img_root_dir,f'{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_root_dir,f'{MP03.ID}_p.mapping'))
MP01.save(filename=os.path.join(img_root_dir,f'{MP01.ID}_p.mapping'))

# %%

#%%
#####Please UPDATE THE METHODS!!!!
MP02._save_nib()
print(MP02.valueList)      
#%%
map_data=sio.loadmat(os.path.join(img_root_dir,'MP02_T2.mat'))
map_data=map_data['T2']
#MP02._map=map_data
for ss in range(MP02.Nz):
    plt.figure()
    plt.imshow(map_data[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    #img_dir= os.path.join(img_root_dir,f'Slice{ss}_T2')
    #plt.savefig(img_dir)
    #MP02.save(filename=os.path.join(img_root_dir,f'{MP02.ID}.mapping'))
# %%
