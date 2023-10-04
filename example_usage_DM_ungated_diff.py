#########################################################################################
#########################################################################################
# CIRC's Diffusion Libraries
#
# Christopher Nguyen, PhD
# Cardiovascular Innovation Research Center (CIRC)
# Cleveland Clinic
#
# 6/2022 Version 1
#########################################################################################
#########################################################################################
#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import pydicom 
from pydicom.filereader import read_dicomdir
#from CIRC_tools import imshowgrid, uigetfile, toc
from pathlib import Path
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tensorly.decomposition import tucker
import tensorly as tl
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
import time
import multiprocessing
import imageio # for gif
from roipoly import RoiPoly, MultiRoi
from matplotlib import pyplot as plt #for ROI poly and croping
from imgbasics import imcrop #for croping
from tqdm.auto import tqdm # progress bar
from ipyfilechooser import FileChooser # ui get file
import pickle # to save diffusion object
import fnmatch # this is for string comparison dicomread
import pandas as pd
from skimage.transform import resize as imresize
import nibabel as nib
import scipy.io as sio
from sklearn import preprocessing

# %% Import libraries ###################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libDiffusion import diffusion  # <--- this is all you need to do diffusion processing

# define class object and read in data
# select the dicom of the folder that holds all dicoms of diffusion data
# OR you can select a previous *.diffusion processed file to open it
# if you don't set ID number it will be taken from DICOM
# processing on your own laptop is default (bMaynard=False)
path=r'C:\Research\MRI\Ungated\CIRC_00325\MR_ep2d_diff_moco2asym_ungated_b500_TE59_32ave'
dti = diffusion(data=path)
dti.cropzone=[]
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 
data_final, diffBVal, diffGrad, datasets=dti.dicomread(dirpath=path)


# %% Crop Data ##########################################################################
#%matplotlib qt <--- you need this if you haven't turned it on in vscode

# this is a simple crop tool just click in the upper left and the lower right of box
crop_data,cropzone=dti._crop(data_final)
#dti.imshow() #show current class data
#%%
#pLOT THE ORIGINAL
import numpy as np
A2=np.copy(crop_data)
k=60
fig,axs=plt.subplots(4,20,figsize=(20,80))
axs=axs.flatten()
Nx,Ny,_,Nd=np.shape(A2)
#Try to find the first direction, and then caculate the kmeans
read_slice=6
X_train_norm = preprocessing.normalize(A2[:,:,read_slice,:].reshape(Nx*Ny,Nd))
for ind,ax in enumerate(axs):
    ax.imshow(X_train_norm[:,ind].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[rieadlist[d]]}ms',fontsize=5)
    ax.axis('off')
plt.savefig(f'CIRC_00325Ori_Nor.jpg')
plt.show()
k=60
fig,axs=plt.subplots(4,20,figsize=(20,80))
axs=axs.flatten()
for ind,ax in enumerate(axs):
    ax.imshow(A2[:,:,read_slice,ind].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[rieadlist[d]]}ms',fontsize=5)
    ax.axis('off')
plt.savefig(f'CIRC_00325Ori.jpg')
plt.show()
#%%
#KMeans in the first direction
import numpy as np
A2=np.copy(crop_data)
Nx,Ny,_,Nd=np.shape(A2)
#Filtered out first direction
from sklearn import preprocessing
from sklearn.cluster import KMeans
X_train_norm = preprocessing.normalize(diffGrad)
kmeans = KMeans(n_clusters = 12, random_state = 0, n_init='auto')
kmeans.fit(X_train_norm)
print(kmeans.cluster_centers_)
b_vector=kmeans.cluster_centers_
indexList=[]
for i in range(len(b_vector)):
    index=np.where(np.dot(diffGrad,b_vector[i]).round(decimals=1)==1)
    indexList.append(index)
#%%
#Try to find the first direction, and then caculate the kmeans
read_slice=6
Nd=int(420/12)
X_train=A2[:,:,read_slice,list(zip(*indexList[0]))].reshape(Nx*Ny,Nd)
b_value=diffBVal[list(zip(*indexList[0]))]
X_train_norm = preprocessing.normalize(X_train).transpose(1,0)
fits = []
score = []
for k in range(2,20):
    kmeans = KMeans(n_clusters = k, random_state = 0, n_init='auto')
    kmeans.fit(X_train_norm)
    fits.append(kmeans)

#%%

k =20
ind=list(range(2,20)).index(k-1)
fig,axs=plt.subplots(2,k-1,figsize=(40,10))
for i in range(k):
    axs[0,i].imshow(fits[ind].cluster_centers_[i,:].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,i].axis('off')
    axs[1,i].imshow(X_train_norm[i,:].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[1,i].axis('off')

#%%
#----------------Try one set first------------------
A2=np.copy(crop_data)
#A2=np.copy(dti._raw_data)
#Try to find the first direction, and then caculate the kmeans
read_slice=6
read_vector=0
#Filtered out first direction
from sklearn import preprocessing
from sklearn.cluster import KMeans
X_train_norm = preprocessing.normalize(diffGrad)

#Try to find the first direction, and then caculate the kmeans
read_slice=6
read_vector=0
Nx,Ny,_,Nd=np.shape(A2)
Nd=int(420/12)
X_train=A2[:,:,read_slice,list(zip(*indexList[read_vector]))].reshape(Nx*Ny,Nd)
b_value=diffBVal[list(zip(*indexList[read_vector]))]
X_train_norm = preprocessing.normalize(X_train).transpose(1,0)
fits = []
score = []
for k in range(2,20):
    kmeans = KMeans(n_clusters = k,  n_init=5)
    kmeans.fit(X_train_norm)
    fits.append(kmeans)

ind=list(range(2,20)).index(k-1)
fig,axs=plt.subplots(2,k-1,figsize=(40,20))
for i in range(k-1):
    axs[0,i].imshow(fits[ind].cluster_centers_[i,:].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,i].axis('off')
    axs[1,i].imshow(X_train_norm[i,:].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[1,i].axis('off')
fig.suptitle(f'Vector{read_vector}')
#%%
#------------------------------Generate a lot:---------------------------
for read_vector in range(4):
    read_slice=6
    Nx,Ny,_,Nd=np.shape(A2)
    Nd=int(420/12)
    X_train=A2[:,:,read_slice,list(zip(*indexList[read_vector]))].reshape(Nx*Ny,Nd)
    b_value=diffBVal[list(zip(*indexList[read_vector]))]
    X_train_norm = preprocessing.normalize(X_train).transpose(1,0)
    #fits = []
    #score = []
    for k in range(6,20,2):
        kmeans = KMeans(n_clusters = k, random_state = 0, n_init=10)
        kmeans.fit(X_train_norm)
        fits.append(kmeans)
        ind=k-1
        fig,axs=plt.subplots(2,k-1,figsize=(40,20))
        for i in range(k-1):
            axs[0,i].imshow(kmeans.cluster_centers_[i,:].reshape(Nx,Ny),cmap='gray')
            #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
            axs[0,i].axis('off')
            axs[1,i].imshow(X_train_norm[i,:].reshape(Nx,Ny),cmap='gray')
            #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
            axs[1,i].axis('off')
        fig.suptitle(f'CIRC_00325Vector{read_vector}_k{k}')
        plt.savefig(f'CIRC_00325Vector{read_vector}_k{k}_2.jpg')
        plt.show()
        plt.close()


#%%
#---------------------------------svd in one dir-----------------------------------
def _regress(data,rank_img=10,rank_diff=1, rank_diff_resp=2,regressindex=2):
    '''
        Perform motion correction
        Inputs:
            * method: 'lrt' (default) or 'naive'
            * rank_img: for lrt correction, default=10
            * rank_diff: for lrt correction, default=1
            * rank_diff_resp: for lrt correction, default=2 
    '''
    print(' ... Performing low rank tensor motion correction')
    data_regress = np.copy(data)
    #define ranks
    if len(np.shape(data))==4:
        Nx,Ny,Nz,Nd=np.shape(data)
    elif len(np.shape(data))==3:
        Nx,Ny,Nd=np.shape(data)
        Nz=1
    for z in tqdm(range(Nz)):
        print(' ...... Slice '+str(z), end='')
        if Nz == 1:
            data = data.reshape(Nx*Ny, Nd)
        else:
            data = data[:,:,z,:].reshape(Nx*Ny, Nd)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_rescale = scaler.fit_transform(data)
        tensor = tl.tensor(data_rescale)
        
        ## Compute image LR representation
        # U1 = nvecs(data, rankImg)
        #U1 = tl.partial_svd(tensor, rank_img)[0]
        Uall=tl.svd_interface(tensor, n_eigenvecs=rank_img)
        U1 = tl.svd_interface(tensor, n_eigenvecs=rank_img)[0]

        ## compute diffusion representation an projections
        # U2_diff = nvecs(data.T, rankBval)
        #U2_diff = tl.partial_svd(tensor, rank_diff)[2]
        U2_diff = tl.svd_interface(tensor, n_eigenvecs=rank_diff_resp)[2][regressindex,:]

        ## Compute respiration + diffusion low dim representation
        # U2_diff_resp = nvecs(data.T, rankBvalResp)
        #U2_diff_resp = tl.partial_svd(tensor, rank_diff_resp)[2].T
        U2_diff_resp = tl.svd_interface(tensor, n_eigenvecs=rank_diff_resp)[2].T

        ## compute image and respiration cores
        # G_diff_resp = np.linalg.pinv(U1).dot( data ).dot(np.linalg.pinv(U2_diff_resp.T))
        G_diff_resp = tl.tenalg.multi_mode_dot( tensor, 
                                    [np.linalg.pinv(U1), np.linalg.pinv(U2_diff_resp)], 
                                    [0,1] )

        ## regress diffusion out
        U2_diff_regress = np.diag( U2_diff.ravel()**(-1) ).dot(  U2_diff_resp )
        # data_diff_regress = U1.dot( G_diff_resp ).dot(U2_diff_regress.T).reshape(data.shape)
        data_diff_regress = tl.to_numpy(
                                tl.tucker_to_tensor( (G_diff_resp, (U1, U2_diff_regress)) )
                                        ).reshape((Nx,Ny,Nd))
        
        return Uall,U2_diff,U2_diff_resp,G_diff_resp,U2_diff_regress,data_diff_regress
data_to_regress=X_train.reshape(Nx,Ny,Nd)
Nx,Ny,Nd=np.shape(data_to_regress)
Uall,_,_,_,_,data_regressed=_regress(data_to_regress,rank_diff_resp=4,regressindex=1)
fig,axes=plt.subplots(2,11,figsize=(20,4))
for i in range(22):
    axes[int(i//11),int(i%11)].imshow(data_regressed[:,:,i],cmap='gray')
    axes[int(i//11),int(i%11)].axis('off')
fig.suptitle('Regress First')

#%%







# %% SINGLE CLIKC SECTION ##################################
# Run this cell and it will do everything in one go 

# 1. Resize
# 2. MoCo
# 3. DTI calc --> window pop open for HA
# 4. save data
#
# 1.  Resize Data ##########################################################################
dti.go_resize(scale=2)

# 2.  Motion Correct Data #################################################################
# you should crop before using mocoas it has the best performance
# it typically takes 2-5min per slice depending on how many directions acquired

# default is LRT but I made it explicit so you can see
# other methods is 'naive' which just registers everything
dti.go_moco(method = 'lrt')

# 3. Calculate DTI #################################################################
# calculates diffusion tensor pixelwise and all the DTI maps
# NB: you NEED to run *.go_segment_LV() first to get helix calculation

# should be relatively fast using C++/Fortran libraries (if its slow then talk to Chris)
dti.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False)
# custom imshow to show the diffusion parameters
dti.imshow_diff_params()

# 4. Save Data #########################################################################
# saves the dti object which basically saves everything such as
# parameters, data, raw_data, maps, ROIs, masks, etc.
# when in doubt don't forget to save!
#
# by default it will save as dti.path + '/' + dti.ID + ".diff" 
dti.save()






####################################################################################################
# If you need to write a paper and you need segmentation 
####################################################################################################

# %% Segment LV #################################################################
# this is using roipoly library so its a bit slow to interact (someday I will rewrite)
# click "new ROI" and then click to connect the vertices
# obviously more verticies the better the outline -- there is NO interpolation between the vertices
# it will go through all the slices
# click finish when you are done

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


