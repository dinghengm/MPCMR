#########################################################################################
#########################################################################################
# CIRC's Diffusion Libraries
#
# Christopher Nguyen, PhD
# Cardiovascular Innovation Research Center (CIRC)
# Cleveland Clinic
#
# 6/2023 Version 1 start port
# 9/2023 Version 1.2 Stable for MD, FA, HA single click analysis
#########################################################################################
#########################################################################################


# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libDiffusion_DCK import diffusion  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
dti = diffusion(r"C:\Users\karad2\OneDrive - Cleveland Clinic\Python Scripts\danielle\dicom_read\CIRC_00192_22737\MR ep2d_diff_Cima_M2_asym_5slices_b500_TE59_FOVphase37.5\MR ep2d_diff_Cima_M2_asym_5slices_b500_TE59_FOVphase37.5.diffusion")

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# %% ####################################################################################
# Crop, Resize, and Motion Correct #############################################################################
#########################################################################################

# Crop
dti.go_crop()

# %% Denoise

# Denoise
dti.go_denoise()

dti.imshow()

# %%
# Resize Data 
dti.go_resize(scale=2)


# %%

# Motion Correct Data 
dti.go_moco(method = 'lrt')

# save
dti.save()

# %% ####################################################################################
# Calculate DTI Params ##################################################################
#########################################################################################

#  Calculate DTI 
dti.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False)

# custom imshow to show the diffusion parameters
dti.imshow_diff_params()

# save
dti.save()


# %% #####################################################################################
# Segment LV #############################################################################
##########################################################################################

#%matplotlib qt <--- you need this if you haven't turned it on in vscode
dti.go_segment_LV(reject=None, image_type="b0_avg")

# save
dti.save()

# look at stats
dti.show_calc_stats_LV()

# look at HA maps and overlays
%matplotlib inline
dti.check_segmentation()

# %% #####################################################################################
# Re-Segment LV as needed ################################################################
##########################################################################################

num_slices = dti.Nz

for sl in range(num_slices):

    # initial plot
    %matplotlib inline
    dti.check_segmentation(sl)
    
    resegment = True
    
    # decide if resegmentation is needed
    print("Perform resegmentation? (Y/N)")
    tmp = input()
    resegment = (tmp == "Y") or (tmp == "y")
    
    if resegment:
        
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
        dti.go_resegment_LV(z=sl, roi_names=roi_names, dilate=True, kernel=3)
        
        # re-plot
        %matplotlib inline
        dti.check_segmentation(sl)
        
        # save
        dti.save() 

        # resegment?
        print("Perform resegmentation? (Y/N)")
        tmp = input()
        resegment = (tmp == "Y") or (tmp == "y")


# %% #####################################################################################
# Export stats #############################################################################
##########################################################################################

dti.export_stats() 
dti.save()


# %% #####################################################################################
# Save Images per slice #############################################################################
##########################################################################################

num_slice = dti.Nz

# Create and save figures - per slice
figsize = (4*3, 3)

for sl in range(num_slice):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, constrained_layout=True)
    
    # MD
    vmin = 0
    vmax = 3e-3 # vmax = np.max(dti.md)
    axes[0].set_axis_off()
    im = axes[0].imshow(dti.md[..., sl], vmin=vmin, vmax=vmax, cmap="hot", interpolation=None)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.95, pad=0.04, aspect=11)
        
    # FA
    vmin = 0
    vmax = 1 # vmax = np.max(dti.fa[dti.mask_lv])
    axes[1].set_axis_off()
    im = axes[1].imshow(dti.fa[..., sl], vmin=vmin, vmax=vmax, cmap="BuPu", interpolation=None)
    cbar = fig.colorbar(im, ax=axes[1], shrink=0.95, pad=0.04, aspect=11)

    # HA
    vmin = -60
    vmax = 60
    axes[2].set_axis_off()
    im = axes[2].imshow(dti.ha[..., sl], vmin=vmin, vmax=vmax, cmap="jet", interpolation=None)
    cbar = fig.colorbar(im, ax=axes[2], shrink=0.95, pad=0.04, aspect=11)

    plt.savefig(os.path.join(dti.path, f"DTI_maps_sl{sl}"))
    plt.show()  


# %% #####################################################################################
# Save Overlay Images per slice #############################################################################
##########################################################################################

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

    plt.savefig(os.path.join(dti.path, f"DTI_maps_sl{sl}"))
    plt.show()  
    
    
####################################################################################################
# OPTIONAL -- helpful and useful stuff below
####################################################################################################

# %% if you want to fix the gradient directions for fixed HA maps
# first try both as False
import numpy as np
bOrient = False # Siemens sometimes flips x and y
bInvert = False # Siemens sometimes inverts y

# you may need to rotate
if bOrient:
    temp = np.copy(dti.bvec[:,0])
    dti.bvec[:,0] = dti.bvec[:,1]
    dti.bvec[:,1] = temp

# or invert depending on the orientation
if bInvert:
    dti.bvec[:,1] *= -1


# %% if you want to visualize motion corrected data 
import numpy as np
max_all = np.max([np.max(dti._raw_data[:]), np.max(dti._data[:])])
A1 = dti._raw_data/max_all #original data
A1,_ = dti._crop(data=A1) #<-- this is meant to be private but I am accessing to show
A1 = dti._resize(data=A1)
A2 = dti._data/max_all #processed data
# you can use dti.imshow to show other volumes besides the class data (like an overloaded fcn)
# here is an example of showing we can send in a custom volume comparing orig data vs moco data
dti.imshow(volume=A1, frameHW=[500,600])
dti.imshow(volume=A2, frameHW=[500,600]) 
dti.imshow(volume=A1-A2, frameHW=[500,600])  
#dti.imshow(volume=np.concatenate((A1,A2,A1-A2),axis=0), frameHW=[500,600]) 

# %% some other usage notes of the "diffusion" class object

# note you can access the parameter maps like this
dti.md              # mean diffusivity [mm^2/s]
dti.fa              # fractional anisotropy
dti.ha              # helix angle [deg]

# there are other things you can access
dti._data           # [Nx,Ny,Nz,Nd] current processed data
dti.Nx              # Number of x pixels
dti.Ny              # Number of y pixels
dti.Nz              # Number of slices
dti.Nd              # Number of diffusion measurements
dti.shape           # shape of the current processed data
dti._data_regress   # from LRT moco
dti._raw_data       # the original data loaded
dti.bvec            # diffusion gradient vectors
dti.bval            # diffusion bvalues



# %% create gif #########################################################

# creates a gif of the diffusion motion corrected data from above
# you can use it to create any gif really just make sure the 2nd argument is a matrix
dti.createGIF('compare_moco.gif',np.concatenate((A1,A2),axis=0)[:,:,2,:], fps=10)

