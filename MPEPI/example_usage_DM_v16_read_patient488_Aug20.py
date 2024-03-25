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
    obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_post',plot=plot,path=saved_img_root_dir)
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
    obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_post_Cropped_updated',plot=plot,path=saved_img_root_dir)

#%%
#488 only
#MP01_0._delete(d=[4,-1])
#MP01_1._delete(d=[2,-1])
MP01_2._delete(d=[6,7])
MP01_0.imshow_corrected(ID=f'MP01_Slice0_2',plot=plot,path=saved_img_root_dir)
MP01_1.imshow_corrected(ID=f'MP01_Slice1_2',plot=plot,path=saved_img_root_dir)
MP01_2.imshow_corrected(ID=f'MP01_Slice2_2',plot=plot,path=saved_img_root_dir)

#%%
#GO_resize and go correct the images:
%matplotlib qt


for obj in [MP01_0,MP01_1,MP01_2,MP02,MP01_post_0,MP01_post_1,MP01_post_2,MP03,MP01,MP01_post]:
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
for ss,obj_T1 in enumerate(MP01_list):
    finalMap,finalRa,finalRb,finalRes=obj_T1.go_ir_fit(searchtype='grid',invertPoint=12)
    plt.figure()
    plt.axis('off')
    plt.imshow(finalMap.squeeze(),cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_T1')
    plt.savefig(img_dir)
    obj_T1._map=finalMap
    #obj_T1.save(filename=os.path.join(img_root_dir,f'{obj_T1.ID}_p.mapping'))
#%%
#Calculate the T1post maps
for ss,obj_T1 in enumerate(MP01_post_list):
    finalMap,finalRa,finalRb,finalRes=obj_T1.go_ir_fit(searchtype='grid',invertPoint=2)
    plt.figure()
    plt.axis('off')
    plt.imshow(finalMap.squeeze(),cmap='magma',vmin=0,vmax=1000)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_T1_post')
    plt.savefig(img_dir)
    obj_T1._map=finalMap
    #obj_T1.save(filename=os.path.join(img_root_dir,f'{obj_T1.ID}_p.mapping'))

#%%

MP02._update()
MP02.imshow_corrected(ID=f'MP02_Cropped',plot=plot,path=img_root_dir)

#%%
MP03._update()
MP03.imshow_corrected(ID=f'MP03_Cropped',valueList=MP03.bval,plot=plot,path=img_root_dir)
MP03.go_cal_ADC()
#%%
MP02.go_t2_fit()


#%%
plt.style.use('default')
for ss in range(MP02.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP02._map[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_T2')
    
    plt.savefig(img_dir)
MP02.save(filename=os.path.join(saved_img_root_dir,f'{MP02.ID}_m.mapping'))
#%%
map_data=np.copy(MP02._map)
map_data[:,:,0]=np.squeeze(MP01_0._map)
map_data[:,:,1]=np.squeeze(MP01_1._map)
map_data[:,:,2]=np.squeeze(MP01_2._map)
MP01._map= np.squeeze(map_data)
MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.ID}_m.mapping'))
#%%
map_data=np.copy(MP02._map)
map_data[:,:,0]=np.squeeze(MP01_post_0._map)
map_data[:,:,1]=np.squeeze(MP01_post_1._map)
map_data[:,:,2]=np.squeeze(MP01_post_2._map)
MP01_post._map= np.squeeze(map_data)
MP01_post.save(filename=os.path.join(saved_img_root_dir,f'{MP01_post.ID}_m.mapping'))

#%%
Nx,Ny,Nz,Nd=np.shape(MP03._data)
for ss in range(MP03.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP03.ADC[:,:,ss],cmap='hot',vmin=0,vmax=3)
    img_dir= os.path.join(saved_img_root_dir,f'Slice{ss}_DWI')
    plt.savefig(img_dir)
MP03.save(filename=os.path.join(saved_img_root_dir,f'{MP03.ID}_m.mapping'))

#%%
%matplotlib qt 

#%%
plt.style.use('default')
def imshowMap(obj,path,plot):
    num_slice=obj.Nz
    volume=obj._map
    ID=str('map_' + obj.CIRC_ID + '_' + obj.ID)
    crange=obj.crange
    cmap=obj.cmap
    figsize = (3.4*num_slice, 3)

    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    axes=axes.ravel()
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(volume[..., sl],vmin=crange[0],vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.018, aspect=8)
    img_dir= os.path.join(path,f'{ID}')
    if plot:
        plt.savefig(img_dir)
    pass
#%%

#img_save_dir=os.path.join(img_root_dir,CIRC_ID)
img_save_dir=saved_img_root_dir
%matplotlib inline
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
imshowMap(obj=MP02,plot=plot,path=img_save_dir)
imshowMap(obj=MP01,plot=plot,path=img_save_dir)
imshowMap(obj=MP03,plot=plot,path=img_save_dir)


#%%
%matplotlib qt
MP02.go_segment_LV(reject=None, image_type="b0_avg",roi_names=['endo', 'epi'])

#%%
MP02.show_calc_stats_LV()
MP03._update_mask(MP02)
MP03.show_calc_stats_LV()

MP01._update_mask(MP02)
MP01.show_calc_stats_LV()

#MP01_post._update_mask(MP02)
#MP01_post.show_calc_stats_LV()
#%%
def testing_plot(obj1,obj2,obj3,obj4,sl):
    
    %matplotlib inline
    alpha = 1.0*obj1.mask_lv[..., sl]

    print(f"Slice {sl}")
    # map map and overlay
    figsize = (4, 2)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize, constrained_layout=True)

    for ind,obj in enumerate([obj1,obj2,obj3,obj4]):
        # map
        crange=obj.crange

        axes[ind,0].set_axis_off()
        im = axes[ind,0].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

        # map overlay
        base_sum=np.array([obj1._data[:, :, sl, 1],obj2._data[:, :, sl, 0],obj3._data[:, :, sl, 0]])
        base_im = np.mean(base_sum,axis=0)
        brightness = 0.8
        axes[ind,1].set_axis_off()
        axes[ind,1].imshow(base_im,vmax=np.max(base_im)*brightness, cmap="gray")
        im = axes[ind,1].imshow(obj._map[..., sl], alpha=alpha, vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

    plt.show()     
    pass
def testing_reseg(obj1,obj2,obj3,obj4):
    numberSlice=obj1.Nz
    obj=obj2
    for sl in range(numberSlice):
        testing_plot(obj1,obj2,obj3,obj4,sl)
        resegment = True
        
        # while resegment:
        # decide if resegmentation is needed
        print("Perform resegmentation? (Y/N)")
        tmp = input()
        resegment = (tmp == "Y") or (tmp == "y")
        
        while resegment:
            
            print("Resegment endo? (Y/N)")
            tmp = input()
            reseg_endo = (tmp == "Y") or (tmp == "y")
            
            print("Resegment epi? (Y/N)")
            tmp = input()
            reseg_epi = (tmp == "Y") or (tmp == "y") 
            
            roi_names = np.array(["endo", "epi"])
            roi_names = roi_names[np.argwhere([reseg_endo, reseg_epi]).ravel()]
            
            %matplotlib qt  
            # selectively resegment LV
            print("Kernel size: ")
            kernel = int(input())
            obj.go_resegment_LV(z=sl, roi_names=roi_names, dilate=True, kernel=kernel,image_type="map")
            
            # re-plot
            testing_plot(obj1,obj2,obj3,obj4,sl)

            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            obj1._update_mask(obj2)
            obj3._update_mask(obj2)
            obj4._update_mask(obj2)
            testing_plot(obj1,obj2,obj3,obj4,sl)
            obj1.show_calc_stats_LV()
            obj2.show_calc_stats_LV()
            obj3.show_calc_stats_LV()
            obj4.show_calc_stats_LV()
    obj1.save(filename=os.path.join(img_save_dir,f'{obj1.CIRC_ID}_{obj1.ID}_p.mapping')) 
    obj2.save(filename=os.path.join(img_save_dir,f'{obj2.CIRC_ID}_{obj2.ID}_p.mapping')) 
    obj3.save(filename=os.path.join(img_save_dir,f'{obj3.CIRC_ID}_{obj3.ID}_p.mapping')) 
    obj4.save(filename=os.path.join(img_save_dir,f'{obj4.CIRC_ID}_{obj3.ID}_p.mapping')) 

    pass

#%%
%matplotlib inline
plt.style.use('default')
testing_reseg(MP01,MP03,MP02,MP03)


# %% View Maps Overlay

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
    plt.savefig(os.path.join(saved_img_root_dir, f"{MP01.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP02.crange
cmap=MP02.cmap
#ase_im = MP02._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
    im = axes[sl].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(saved_img_root_dir, f"{MP02.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP03.crange
cmap=MP03.cmap
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
    im = axes[sl].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(saved_img_root_dir, f"{MP03.ID}_overlay_maps.png"))
plt.show()  
'''
#T1POST
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
MP01_post.crange=[0,1600]
crange=MP01_post.crange
cmap=MP01_post.cmap
#base_im = MP01_post._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.8))
    im = axes[sl].imshow(MP01_post._map[..., sl], vmin=crange[0], vmax=crange[1], cmap='magma', alpha=1.0*MP01_post.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{MP01_post.ID}_overlay_maps.png"))
plt.show()  
'''
#%%
MP02.show_calc_stats_LV()
MP03.show_calc_stats_LV()
MP01.show_calc_stats_LV()
#MP01_post.show_calc_stats_LV()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3])
MP01_post.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,1600])

# %%
MP01_0.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_0.ID}_p.mapping'))
MP01_1.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_1.ID}_p.mapping'))
MP01_2.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_2.ID}_p.mapping'))
MP01_post_0.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_post_0.ID}_p.mapping'))
MP01_post_1.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_post_1.ID}_p.mapping'))
MP01_post_2.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_post_2.ID}_p.mapping'))


MP01_post.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01_post.ID}_p.mapping'))
MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))
MP02.save(filename=os.path.join(saved_img_root_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(saved_img_root_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))

# %%
