#%%
%matplotlib qt                      
from libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v12 import readFolder,decompose_LRT,go_ir_fit,moco,moco_naive
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
plt.rcParams['savefig.dpi'] = 500
from t1_fitter import T1_fitter,go_fit_T1
plt.rcParams.update({'axes.titlesize': 'small'})
#%%
plot=True
# %%
CIRC_ID='CIRC_00405'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
dirpath=os.path.dirname(dicomPath)
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
# %%
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.go_moco()
MP03.imshow_corrected(ID=f'{MP03.ID}_raw',plot=plot)
cropzone=MP03.cropzone
ADC=MP03.go_cal_ADC()
MP03._map=ADC*1000
#%%
MP03.imshow_map()
 



#%%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI_DELAY50')
dirpath=os.path.dirname(dicomPath)
MP03_50 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
# %%
MP03_50.cropzone=cropzone
MP03_50.go_crop()
MP03_50.go_resize(scale=2)
MP03_50.go_moco()
#%%
MP03_50.imshow_corrected(ID=f'{MP03_50.ID}_raw',plot=plot)
ADC=MP03_50.go_cal_ADC()
MP03_50._map=ADC*1000
MP03_50.imshow_map()
# %%
figsize = (4*3, 3)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, constrained_layout=True)
for sl in range(3):
    # MD
    vmin = 0
    vmax = 3 # vmax = np.max(dti.md)
    axes[sl].set_axis_off()
    im=axes[sl].imshow(MP03_50._map[..., sl], vmin=vmin, vmax=vmax, cmap="hot")
cbar = fig.colorbar(im, ax=axes[2], shrink=1, pad=0.04, aspect=11)
plt.savefig(os.path.join(dirpath,f'{MP03_50.ID}_map'))
# %%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI_DELAY50_Standard')
dirpath=os.path.dirname(dicomPath)
MP03_50_Std = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
# %%
MP03_50_Std.cropzone=cropzone
MP03_50_Std.go_crop()
MP03_50_Std.go_resize(scale=2)
MP03_50_Std.go_moco()
#%%
MP03_50_Std.imshow_corrected(ID=f'{MP03_50_Std.ID}_raw',plot=plot)
ADC=MP03_50_Std.go_cal_ADC()
MP03_50_Std._map=ADC*1000
MP03_50_Std.imshow_map()
# %%

%matplotlib inline                        
from libDiffusion_V2 import diffusion  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 
path=r'C:\Research\MRI\MP_EPI\CIRC_00405_22737_CIRC_00405_22737\CIRC_RESEARCH CIRC Research\MR ep2d_diff_Cima_M2_asym_5slices_b500_TE59_FOVphase37.5-2'
dti = diffusion(data=path)

#%%
# click in the upper left then lower right to crop
%matplotlib qt
dti.cropzone=cropzone
dti.go_crop()

# show cropped data
dti.imshow() 
# %%
# Resize, Register, and Calculate DTI Parameters ##################################

# Resize Data 
dti.go_resize(scale=2)

# Motion Correct Data 
dti.go_moco(method = 'lrt')
#%%
# Calculate DTI 
# NB: you NEED to run *.go_segment_LV() first to get helix calculation
%matplotlib qt
dti.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False)

# custom imshow to show the diffusion parameters
dti.imshow_diff_params()

# Save Data (by default it will save as dti.path + '/' + dti.ID + ".diff") 
dti.save()
# %%
CIRC_ID = "CIRC_00373"
root_dir = r"C:\Research\MRI\MP_EPI"

%matplotlib inline

img_dir = os.path.join(root_dir, "saved_ims",dti.ID)
# make directory for saved images if it doesn't exist yet
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)
num_slice = 5
num_dif = 12
# number of averages (b=50 + b=500)
num_avg = 9

# expected number of dicom files if scan finished
num_if_finished = num_slice * num_dif * num_avg
figsize = (3.4*num_slice, 3)

# MD
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
vmin = 0
vmax = 5e-3 # vmax = np.max(dti.md)
for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(dti.md[..., sl], vmin=vmin, vmax=vmax, cmap="hot")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(os.path.join(img_dir, "MD_maps"))
plt.show()
    
# FA
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
vmin = 0
vmax = 0.5 # vmax = np.max(dti.fa[dti.mask_lv])
for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(dti.fa[..., sl], vmin=vmin, vmax=vmax, cmap="BuPu")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
plt.savefig(os.path.join(img_dir, "FA_maps"))
plt.show()

# HA
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
vmin = -90
vmax = 90
for sl in range(num_slice):
    axes[sl].set_axis_off()
    im = axes[sl].imshow(dti.ha[..., sl], vmin=vmin, vmax=vmax, cmap="jet")
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.97, pad=0.018, aspect=11)
plt.savefig(os.path.join(img_dir, "HA_maps"))
plt.show()  
    

# %% Create and save figures - per slice

%matplotlib inline

img_dir = os.path.join(root_dir, "saved_ims",dti.ID)
# make directory for saved images if it doesn't exist yet
if not os.path.isdir(img_dir):
    os.mkdir(img_dir)

figsize = (4*3, 3)


for sl in range(num_slice):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, constrained_layout=True)
    
    # MD
    vmin = 0
    vmax = 5e-3 # vmax = np.max(dti.md)
    axes[0].set_axis_off()
    im = axes[0].imshow(dti.md[..., sl], vmin=vmin, vmax=vmax, cmap="hot")
    cbar = fig.colorbar(im, ax=axes[0], shrink=1, pad=0.04, aspect=11)
        
    # FA
    vmin = 0
    vmax = 0.5 # vmax = np.max(dti.fa[dti.mask_lv])
    axes[1].set_axis_off()
    im = axes[1].imshow(dti.fa[..., sl], vmin=vmin, vmax=vmax, cmap="BuPu")
    cbar = fig.colorbar(im, ax=axes[1], shrink=1, pad=0.04, aspect=11)

    # HA
    vmin = -90
    vmax = 90
    axes[2].set_axis_off()
    im = axes[2].imshow(dti.ha[..., sl], vmin=vmin, vmax=vmax, cmap="jet")
    cbar = fig.colorbar(im, ax=axes[2], shrink=1, pad=0.04, aspect=11)

    plt.savefig(os.path.join(img_dir, f"DTI_maps_sl{sl}"))
    plt.show()  
    
#%%
#%%
%matplotlib qt
dti.go_segment_LV()

dti.save()
# %%
# dti.export_stats(filename=r'C:\Research\MRI\MP_EPI\dti.csv') 

num_slice = dti.Nz

# Create and save figures - per slice
figsize = (4*3, 3)

for sl in range(num_slice):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, constrained_layout=True)
    alpha = dti.mask_lv[..., sl] * 1.0
    b50_inds = np.argwhere(dti.bval == 50).ravel()
    base_im = np.mean(dti._data[:, :, sl, b50_inds], axis=-1)
    brightness = 0.8
    
    # MD
    vmin = 0
    vmax = 3e-3 # vmax = np.max(dti.md)
    axes[0].set_axis_off()
    axes[0].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
    im = axes[0].imshow(dti.md[..., sl], alpha=alpha, vmin=vmin, vmax=vmax, cmap="hot", interpolation=None)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.95, pad=0.04, aspect=11)
        
    # FA
    vmin = 0
    vmax = 1 # vmax = np.max(dti.fa[dti.mask_lv])
    axes[1].set_axis_off()
    axes[1].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
    im = axes[1].imshow(dti.fa[..., sl], alpha=alpha, vmin=vmin, vmax=vmax, cmap="BuPu", interpolation=None)
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.95, pad=0.04, aspect=11)

    # HA
    vmin = -60
    vmax = 60
    axes[2].set_axis_off()
    axes[2].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
    im = axes[2].imshow(dti.ha[..., sl], alpha=alpha, vmin=vmin, vmax=vmax, cmap="jet", interpolation=None)
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.95, pad=0.04, aspect=11)

    plt.savefig(os.path.join(img_dir, f"DTI_maps_sl{sl}_overlay"))
    plt.show()  
    
    
# %%
dti.export_stats(filename=r'C:\Research\MRI\MP_EPI\dti.csv') 
# %%
