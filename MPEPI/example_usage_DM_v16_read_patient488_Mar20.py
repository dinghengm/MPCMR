# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib inline                     
from libMapping_v13 import *  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
#%%
boolenTest=input('Would you like to save you Plot? "Y" and "y" to save')
if boolenTest.lower() == 'y':
    plot=True
else:
    plot=False


# %%
#Please try to change to CIRC_ID
CIRC_ID='CIRC_00488'
img_root_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024",'CIRC_00488_1')
if not os.path.exists(saved_img_root_dir):
    os.mkdir(saved_img_root_dir)

#img_root_dir = saved_img_root_dir
# image root directory
# Statas saved 
stats_file = os.path.join(defaultPath, "MPEPI_stats_v16.csv") 


#Read the MP01-MP03
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('.mapping'):
            if not path.endswith('m.mapping') and not path.endswith('p.mapping'):
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
MP01_post_0=mapping(mapList[0])
MP01_post_1=mapping(mapList[1])
MP01_post_2=mapping(mapList[2])
MP01_0=mapping(mapList[3])
MP01_1=mapping(mapList[4])
MP01_2=mapping(mapList[5])
MP01=mapping(mapList[6])
MP01_post=mapping(mapList[7])
MP02=mapping(mapList[8])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[9])


#%%
MP01_list=[MP01_0,MP01_1,MP01_2]
MPs_list=[MP01,MP02,MP03]

for obj in MPs_list:
    obj._data=np.copy(obj._raw_data)
for obj in MP01_list:
    obj._data=np.copy(obj._raw_data)
endList=[]
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
            #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped',plot=plot,path=saved_img_root_dir)
            #obj_T1.go_create_GIF(path_dir=str(saved_img_root_dir))
            obj_T1._update_data(moco_data_single_slice[:,:,0:Ndtmp_end])
            print(obj_T1.ID,np.shape(obj_T1._data))
            print('valueList=',obj_T1.valueList)
        else:
            Ndtmp_start=Ndtmp_end
            Ndtmp_end+=np.shape(obj._data)[-1]
            obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
            print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
            #print('valueList=',obj.valueList)
    endList.append(Ndtmp_end)
#%%
#get the data of MP01_post
MP01_post_list=[MP01_post_0,MP01_post_1,MP01_post_2]

for obj in MP01_post_list:
    obj._update_data(obj._raw_data)

for ss,obj_T1 in enumerate(MP01_post_list):
    key=f'moco_Slice{ss}'
    moco_data_single_slice=np.transpose(moco_data[key],(2,1,0))
    Ndtmp_start=endList[ss]
    obj_T1._update_data(moco_data_single_slice[:,:,np.newaxis,Ndtmp_start::])
    obj_T1.go_crop_Auto()
    #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_post',plot=plot,path=saved_img_root_dir)
    #obj_T1.go_create_GIF(path_dir=str(saved_img_root_dir))
    obj_T1._update_data(moco_data_single_slice[:,:,Ndtmp_start::])
    print(obj_T1.ID,np.shape(obj_T1._data))
    print('valueList=',obj_T1.valueList)    #print('valueList=',obj.valueList)

#%%
for ss,obj_T1 in enumerate(MP01_list):
    valueArray=np.array(obj_T1.valueList)
    arrayInd=np.where(np.logical_and(valueArray>=700,valueArray<=2000))
    obj_T1._data[...,arrayInd]=obj_T1._raw_data[...,arrayInd]
    #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_updated',plot=plot,path=saved_img_root_dir)
#%%
MP01_post_0.valueList= [110.0, 190.0, 440.0, 940.0, 1240.0, 2140.0, 3140.0]
for ss,obj_T1 in enumerate(MP01_post_list):
    valueArray=np.array(obj_T1.valueList)
    arrayInd=np.where(np.logical_and(valueArray>=300,valueArray<=700))
    obj_T1._data[...,arrayInd]=obj_T1._raw_data[...,arrayInd]
    #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_post_Cropped_updated',plot=plot,path=saved_img_root_dir)

#%%
#488 only
#MP01_0._delete(d=[4,-1])
#MP01_1._delete(d=[2,-1])
#MP01_2._delete(d=[6,7])
#MP01_0.imshow_corrected(ID=f'MP01_Slice0_2',plot=plot,path=saved_img_root_dir)
#MP01_1.imshow_corrected(ID=f'MP01_Slice1_2',plot=plot,path=saved_img_root_dir)
MP01_2.imshow_corrected(ID=f'MP01_Slice2_2',plot=plot,path=saved_img_root_dir)

#####
#Apply fitting on 3 maps:
#Without #6
import copy 
MP01_2try_1=copy.deepcopy(MP01_2)
MP01_2try_1._delete(d=[6])
MP01_2try_1.imshow_corrected(ID=f'MP01_Slice2_try1',plot=plot,path=saved_img_root_dir)

#Without #6 and -1
MP01_2try_2=copy.deepcopy(MP01_2)
MP01_2try_2._delete(d=[6,-1])
MP01_2try_2.imshow_corrected(ID=f'MP01_Slice2_try2',plot=plot,path=saved_img_root_dir)

#Without #6,7, -1
MP01_2try_3=copy.deepcopy(MP01_2)
MP01_2try_3._delete(d=[6,7,-1])
MP01_2try_3.imshow_corrected(ID=f'MP01_Slice2_try3',plot=plot,path=saved_img_root_dir)







#%%
#GO_resize and go correct the images:
%matplotlib qt


for obj in [MP01_2try_1,MP01_2try_2,MP01_2try_3]:
    obj.go_crop_Auto()
    obj.go_resize()
    obj._update()

#%%
#---...-----
#Please calculate the maps in a loop
#%%
#Calculate the T1 maps
%matplotlib qt
plt.style.use('default')
for ss,obj_T1 in enumerate([MP01_2try_1,MP01_2try_2,MP01_2try_3]):
    finalMap,finalRa,finalRb,finalRes=obj_T1.go_ir_fit(searchtype='grid',invertPoint=12)
    plt.figure()
    plt.axis('off')
    plt.imshow(finalMap.squeeze(),cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(saved_img_root_dir,f'map_MP01_Slice2_try{ss}')
    plt.savefig(img_dir)
    obj_T1._map=finalMap
    obj_T1.save(filename=os.path.join(saved_img_root_dir,f'T1_{ss}_m.mapping'))
#%%
#Calculate the T1post maps

del MP01,MP01_0,MP01_1
MP01=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488\CIRC_00488_MP01_T1_p.mapping')
MP02_contour=mapping(r'C:\Research\MRI\MP_EPI\saved_ims_v2_Feb_5_2024\CIRC_00488\CIRC_00488_MP02_T2_p.mapping')

#%%
for obj in [MP01,MP02,MP03]:
    obj.go_crop_Auto()
    obj.go_resize()
    obj._update()
for ss,obj_T1 in enumerate([MP01_2try_1,MP01_2try_2,MP01_2try_3]):
    MP01._map[:,:,2]=np.squeeze(obj_T1._map)
    #img_save_dir=os.path.join(img_root_dir,CIRC_ID)
    img_save_dir=saved_img_root_dir
    %matplotlib inline
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir) 
    MP01._update_mask(MP02_contour)
    MP01.show_calc_stats_LV()
    #View Maps Overlay
    %matplotlib inline
    # view HAT mask
    num_slice = MP01.Nz 
    figsize = (3.4*num_slice, 3)
    # T1
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=MP01.crange
    cmap=MP01.cmap
    #base_sum=np.array([MP02._data[:, :, :, 0:6],MP03._data[:, :, :, 0:6]])
    #base_im = np.mean(base_sum,axis=0)
    base_sum=np.concatenate([MP02._data[:, :, :, 0:6],MP03._data[:, :, :,  0:12]],axis=-1)
    base_im = np.mean(base_sum,axis=-1)
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
        im = axes[sl].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
    #cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
    if plot:
        plt.savefig(os.path.join(saved_img_root_dir, f"MP01_Slice2_try{ss}_overlay_maps.png"))
    plt.show()  
    plt.close()
    #%%
    MP01.show_calc_stats_LV()
#MP01_post.show_calc_stats_LV()
MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))

# %%
