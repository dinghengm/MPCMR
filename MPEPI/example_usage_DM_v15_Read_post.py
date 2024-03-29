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
import h5py
import pandas as pd
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
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Jan_12_2024",CIRC_ID)
if not os.path.exists(saved_img_root_dir):
    os.mkdir(saved_img_root_dir)
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
#See the moco methods
img_root_dir=saved_img_root_dir
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
        #mp_tmp.createGIF(img_path,A2,fps=5)
        #mp_tmp.imshow_corrected(volume=data_tmp[:,:,np.newaxis,:],valueList=range(1000),ID=f'Slice{ind-3}_Yuchi_moco',plot=plot,path=img_root_dir)

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
    #obj.go_create_GIF(path_dir=str(img_root_dir))
    #obj.go_crop_Auto()
    #obj.go_resize()
#%%
#Save the file as M
stats_file = os.path.join(os.path.dirname(saved_img_root_dir), "MPEPI_stats_v2.csv") 

for obj in MPs_list:
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
%matplotlib inline
#Crop both:
MP02.go_crop_Auto()
#cropzone=MP02.cropzone
MP02.go_resize()
for obj in [MP01,MP03,MP01_post]:
    obj.go_crop_Auto()
    #obj.go_crop()
    obj.go_resize()
    obj._update()

for obj in MPs_list:
    if 'mp03' in obj.ID.lower():
            obj.imshow_corrected(ID=f'{obj.ID}_moco',plot=False,path=saved_img_root_dir,valueList=obj.bval)
    else:

        obj.imshow_corrected(ID=f'{obj.ID}_moco',plot=False,path=saved_img_root_dir)

#%%
MP01._delete(d=[5])
MP01_post._delete([3])
for obj in MPs_list:
    if 'mp03' in obj.ID.lower():
            obj.imshow_corrected(ID=f'{obj.ID}_moco_2',plot=False,path=saved_img_root_dir,valueList=obj.bval)
    else:

        obj.imshow_corrected(ID=f'{obj.ID}_moco_2',plot=False,path=saved_img_root_dir)

#%%

#---...-----
#Please calculate the maps in a loop
finalMap,finalRa,finalRb,finalRes=MP01.go_ir_fit(searchtype='grid',invertPoint=8)
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
#%% Plot

img_save_dir=img_root_dir
MP01_post.crange=[0,1600]
for map in [MP01,MP02,MP03,MP01_post]:
    # create images images per map type
    num_slice = map.Nz
    figsize = (3.4*num_slice, 3)
    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    crange=map.crange
    cmap=map.cmap

    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(map._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.018, aspect=18)
    plt.savefig(os.path.join(img_save_dir, f"{map.CIRC_ID}_{map.ID}"))
    plt.show()
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
imshowMap(obj=MP01_post,plot=plot,path=img_save_dir)

#%%
%matplotlib qt
MP02.go_segment_LV(reject=None, image_type="b0_avg",roi_names=['endo', 'epi','septal', 'lateral'])

#%%
MP02.show_calc_stats_LV()
MP03._update_mask(MP02)
MP03.show_calc_stats_LV()

MP01._update_mask(MP02)
MP01.show_calc_stats_LV()

MP01_post._update_mask(MP02)
MP01_post.show_calc_stats_LV()
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
img_save_dir=saved_img_root_dir
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
    plt.savefig(os.path.join(img_save_dir, f"{MP01.ID}_overlay_maps.png"))
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
    plt.savefig(os.path.join(img_save_dir, f"{MP02.ID}_overlay_maps.png"))
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
    plt.savefig(os.path.join(img_save_dir, f"{MP03.ID}_overlay_maps.png"))
plt.show()  

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


#%%
MP02.show_calc_stats_LV()
MP03.show_calc_stats_LV()
MP01.show_calc_stats_LV()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3])
MP01_post.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,1600])


# %%
MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))
MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))
MP01_post.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP01_post.ID}_p.mapping'))
# %%
