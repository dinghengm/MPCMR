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
CIRC_ID='CIRC_00438'
img_root_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Jan_12_2024",CIRC_ID)
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
        #mp_tmp.createGIF(img_path,A2,fps=5)
        #mp_tmp.imshow_corrected(volume=data_tmp[:,:,np.newaxis,:],valueList=range(1000),ID=f'Slice{ind-3}_Yuchi_moco',plot=plot,path=saved_img_root_dir)
        #mp_tmp.go_crop_Auto()
        #mp_tmp.go_resize()
        #mp_tmp.imshow_corrected(volume=data_tmp[:,:,np.newaxis,:],valueList=range(1000),ID=f'Slice{ind-3}_Yuchi_moco_cropped',plot=plot,path=saved_img_root_dir)
    except:

        pass
#%%


MP01=mapping(mapList[0])

MP02=mapping(mapList[1])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[2])


#%%

MPs_list=[MP01,MP02,MP03,]

for obj in MPs_list:
    obj._data=np.copy(obj._raw_data)
ind_label=[0,MP01.Nd,MP01.Nd+MP02.Nd,MP01.Nd+MP02.Nd+MP03.Nd]
print(ind_label)

for ss in range(3):
    key=f'moco_Slice{ss}'
    moco_data_single_slice=np.transpose(moco_data[key],(2,1,0))

    MPs_list[0]._data[:,:,ss,:]=moco_data_single_slice[:,:,ind_label[0]:ind_label[1]]
    MPs_list[1]._data[:,:,ss,:]=moco_data_single_slice[:,:,ind_label[1]:ind_label[2]]
    MPs_list[2]._data[:,:,ss,:]=moco_data_single_slice[:,:,ind_label[2]:ind_label[3]]
    print(np.shape(MPs_list[0]._data[:,:,ss,:]))
    print(np.shape(MPs_list[1]._data[:,:,ss,:]))
    print(np.shape(MPs_list[2]._data[:,:,ss,:]))
#%%
valueArray=np.array(MP01.valueList)
arrayInd=np.where(np.logical_and(valueArray>=700,valueArray<=1500))
MP01._data[...,arrayInd]=MP01._raw_data[...,arrayInd]
MP01.imshow_corrected(ID=f'MP01_Slice_Cropped_updated',plot=plot,path=saved_img_root_dir)

#%%
#Save the file as M
stats_file = os.path.join(os.path.dirname(saved_img_root_dir), "MPEPI_stats_v2.csv") 


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
#GO_resize and go correct the images:
%matplotlib qt

for obj in [MP01,MP02,MP03]:
    obj.go_crop_Auto()
    obj.go_resize()
    obj._update()
#%%
MP01.imshow_corrected(ID=f'MP01_Slice_Cropped_updated',plot=plot,path=saved_img_root_dir)

#%%
#---...-----
#Please calculate the maps in a loop
#%%
#Calculate the T1 maps
%matplotlib qt
MP01.go_ir_fit()

MP02.imshow_corrected(ID=f'MP02_Cropped',plot=plot,path=img_root_dir)
MP02.go_t2_fit()
MP03.imshow_corrected(ID=f'MP03_Cropped',valueList=MP03.bval,plot=plot,path=img_root_dir)
MP03.go_cal_ADC()
#%%



#%%
for ss in range(MP02.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP02._map[:,:,ss],cmap='viridis',vmin=0,vmax=150)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_T2')
    
    plt.savefig(img_dir)
MP02.save(filename=os.path.join(img_root_dir,f'{MP02.ID}_m.mapping'))
#%%
for ss in range(MP01.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP01._map[:,:,ss],cmap='magma',vmin=0,vmax=3000)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_T1')
    
    plt.savefig(img_dir)
MP01.save(filename=os.path.join(img_root_dir,f'{MP01.ID}_m.mapping'))

#%%
Nx,Ny,Nz,Nd=np.shape(MP03._data)
for ss in range(MP03.Nz):
    plt.figure()
    plt.axis('off')
    plt.imshow(MP03.ADC[:,:,ss],cmap='hot',vmin=0,vmax=3)
    img_dir= os.path.join(img_root_dir,f'Slice{ss}_DWI')
    plt.savefig(img_dir)
MP03.save(filename=os.path.join(img_root_dir,f'{MP03.ID}_m.mapping'))

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


#%%
MP02.show_calc_stats_LV()
MP03.show_calc_stats_LV()
MP01.show_calc_stats_LV()
#MP01_post.show_calc_stats_LV()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3])

# %%
MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))
MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))

# %%
