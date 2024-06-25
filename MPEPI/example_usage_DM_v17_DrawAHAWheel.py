#%%
#This is the file to draw AHA wheel.
#The data in from ims_v2_Feb_5_2024_WITH8000 -> ims_v3_June5_WITH8000


from libMapping_v14 import mapping
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=False
defaultPath= r'C:\Research\MRI\MP_EPI'

#%%
###Hey this won't work with disease!!!
CIRC_ID_List=[446,452,429,419,398,382,381,373,472,498,500]
#CIRC_NUMBER=446
CIRC_NUMBER=CIRC_ID_List[10]
CIRC_ID=f'CIRC_00{int(CIRC_NUMBER)}'
print(f'Running{CIRC_ID}')

img_root_dir = os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000",CIRC_ID)
img_save_dir=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000",CIRC_ID)

if os.path.exists(img_save_dir) is False:
    os.makedirs(img_save_dir)
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            mapList.append(path)
MP01=mapping(mapList[0])
MP02=mapping(mapList[1])
MP03=mapping(mapList[2])
mask=MP02.mask_lv


#%%
plot=False
Nz=MP01.Nz
Nx=np.shape(MP01._map)[0]
Ny=np.shape(MP01._map)[1]
%matplotlib inline

overlay_save_dir=os.path.join(os.path.dirname(img_root_dir),'overlay')

num_slice = MP01.Nz 
figsize = (3.4*num_slice, 3)

# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP01.crange
cmap=MP01.cmap
base_sum=np.array([MP02._data[:, :, :, 0:6],MP03._data[:, :, :, 0:6]])
base_im = np.mean(base_sum,axis=0)
base_im = np.mean(base_im,axis=-1)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP01._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP01.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(overlay_save_dir, f"{MP01.CIRC_ID}_{MP01.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP02.crange
cmap=MP02.cmap
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP02._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP02.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(overlay_save_dir, f"{MP02.CIRC_ID}_{MP02.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=MP03.crange
cmap=MP03.cmap
#base_im = MP03._data[..., 0]

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*0.7))
    im = axes[sl].imshow(MP03._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*MP03.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(overlay_save_dir, f"{MP03.CIRC_ID}_{MP03.ID}_overlay_maps.png"))
plt.show()  

#%%
#Get the center
%matplotlib qt
MP01.go_define_CoMandRVIns()
MP01.go_AHA_wheel_check()
#%%
#Copy the map from T1 T2 and ADC
MP01.go_get_AHA_wheel()

MP02._update_mask(MP01)
MP03._update_mask(MP01)
segment_16=MP02.segment_16
maskFinal=np.zeros((Nx,Ny),dtype=int)

#%%
plot=True
figsize = (3.4*Nz, 3)
fig, axes = plt.subplots(nrows=1, ncols=Nz, figsize=figsize, constrained_layout=True)
axes=axes.ravel()
segment_16=MP01.segment_16

for nn in range(Nz):
    axes[nn].set_axis_off()
    
    if nn<2:
        maskFinal=np.zeros((Nx,Ny),dtype=int)    
        segmentNum =6   
        
        for seg in range(segmentNum):
            maskFinal+=segment_16[nn][seg] * (seg +1)
            
        im = axes[nn].imshow(maskFinal,vmax=6)
    else:
        maskFinal=np.zeros((Nx,Ny),dtype=int)
        segmentNum=4
        #The last slice
        for seg in range(segmentNum):
            maskFinal+=segment_16[nn][seg] * (seg +1)
            
        im = axes[nn].imshow(maskFinal,vmax=4)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"AHA_16_segments.png"))
        
# %%
filename=os.path.join(os.path.dirname(img_save_dir),'mapping_AHA.csv')
MP01.export_stats(filename=filename,crange=[0,1800])
MP02.export_stats(filename=filename,crange=[0,150])
MP03.export_stats(filename=filename,crange=[0,3])

# %%
MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))
MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))

# %%
