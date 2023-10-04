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
from libDiffusion import diffusion  # <--- this is all you need to do diffusion processing

# define class object and read in data
# select the dicom of the folder that holds all dicoms of diffusion data
# OR you can select a previous *.diffusion processed file to open it
# if you don't set ID number it will be taken from DICOM
# processing on your own laptop is default (bMaynard=False)
dti = diffusion()

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data_final, diffBVal, diffGrad, datasets=dti.dicomread(dirpath=r'C:\Research\MRI\Ungated\CIRC_00325\MR_ep2d_diff_Cima_M2_asym_5slices_b500_TE59_FOVphase37.5')


# %% ####################################################################################
# Crop Data #############################################################################
#########################################################################################
# this is a simple crop tool just click in the upper left and the lower right of box
# we will replace with ML tool (looking at you Hoa :p )
dti.go_crop()


# %% ####################################################################################
# SINGLE CLICK SECTION ##################################################################
#########################################################################################

# Run this cell and it will do the following in one go:
# 1. Resize
# 2. MoCo
# 3. DTI calc --> window pop open for HA
# 4. save data
#
# 1.  Resize Data #######################################################################
dti.go_resize(scale=2)

# 2.  Motion Correct Data ###############################################################
# you should crop before using mocoas it has the best performance
# it typically takes 2-5min per slice depending on how many directions acquired

# default is LRT but I made it explicit so you can see
# other methods is 'naive' which just registers everything
dti.go_moco(method = 'lrt')

# 3. Calculate DTI ######################################################################
# calculates diffusion tensor pixelwise and all the DTI maps
# NB: you NEED to run *.go_segment_LV() first to get helix calculation

# should be relatively fast using C++/Fortran libraries (if its slow then talk to Chris)
dti.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False)
# custom imshow to show the diffusion parameters
dti.imshow_diff_params()

# 4. Save Data ###########################################################################
# saves the dti object which basically saves everything such as
# parameters, data, raw_data, maps, ROIs, masks, etc.
# when in doubt don't forget to save!
#
# by default it will save as dti.path + '/' + dti.ID + ".diff" 
dti.save()


# %% #####################################################################################
# Segment LV #############################################################################
##########################################################################################
# this is using roipoly library so its a bit slow to interact (someday I will rewrite)
# click "new ROI" and then click to connect the vertices
# obviously more verticies the better the outline --> there is NO interpolation 
# between the vertices
# it will go through all the slices and click finish when you are done for each slice

#%matplotlib qt <--- you need this if you haven't turned it on in vscode
dti.go_segment_LV()
dti.export_stats() #not fullly tested (will test with Lily)
dti.save()






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


