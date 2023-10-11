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
from libMapping_v12 import mapping,readFolder
import numpy as np
import matplotlib.pyplot as plt
import os
# %%
path=r'C:\Research\MRI\exvivo\UCSF_7020_05162023\20230515_102113_900000_1\Beat_Perf_GRE'
import fnmatch,pydicom
import SimpleITK as sitk
dicom_filelist = fnmatch.filter(sorted(os.listdir(path)),'*.dcm')
if dicom_filelist == []:
    dicom_filelist = fnmatch.filter(sorted(os.listdir(path)),'*.DCM')
datasets = [pydicom.dcmread(os.path.join(path, file))
                        for file in dicom_filelist]
img = datasets[0].pixel_array
Nx, Ny,Nz = img.shape
Nd = len(datasets)
data = np.zeros((Nx,Ny,Nz,Nd))
sliceLocsArray = []
diffGradArray = []
diffBValArray = []
for ind,ds in enumerate(datasets):
    data[:,:,:,ind]=ds.pixel_array
    try:
        diffGrad_str=ds['0x5200','0x9230'][0]['0x0018','0x9117'][0]['0x0018','0x9076'][0]['0x0018','0x9089'].value
    except:
        diffGrad_str=[0,0,0]
    diffGradArray.append(diffGrad_str)
    diffBVal_str=ds['0x5200','0x9230'][0]['0x0018','0x9117'][0]['0x0018','0x9087']
    diffBValArray.append(diffBVal_str.value)
# %%
#Maybe I assume 1 b0, 72 b500, and it's 12 Direction * 6
path=r'C:\Research\MRI\Ungated\CIRC_00325\MR_ep2d_diff_moco2asym_ungated_b500_TE59_32ave'
dti = diffusion(data=path)

# %%
data=data.transpose(1,2,0,3)
dti._data=data
dti.bval=diffBValArray
dti.bvec=np.array(diffGradArray)
Nx,Ny,Nz,Nd=np.shape(data)
dti.Nx=Nx
dti.Ny=Ny
dti.Nz=Nz
dti.Nd=Nd
dti.shape=np.shape(data)

#%%
#Plot the single images
plt.subplot(211)
plt.imshow(dti._data[:,:,40,0],cmap='gray')
plt.subplot(212)
plt.imshow(dti._data[:,:,40,1],cmap='gray')
# %%
def setThrehold(data=None,thresholdcommand=2):
    #calculate the Y diviation:
    #Reshape
    Nx,Ny,Nz,Nd=np.shape(data)
    Y=data.reshape((Nx*Ny,Nz,Nd))
    Y_mask=np.ones((Nx*Ny,Nz,Nd),dtype=bool)
    threshold = thresholdcommand*np.std(Y,axis=0)
    Y_mask[abs(Y)<threshold]=0

    return Y_mask.reshape((Nx,Ny,Nz,Nd))
#%%
#################TODO see what is kernel PCA
Nx,Ny,Nz,Nd=np.shape(data)
Y=data.reshape((Nx*Ny,Nz,Nd))
Y_mask=np.ones((Nx*Ny,Nz,Nd),dtype=bool)
Y_mask[abs(Y)>1000]=0
Y_mask[abs(Y)<50]=0
datatemp=Y*Y_mask
from sklearn.decomposition import KernelPCA
kernel_pca = KernelPCA(
    n_components=5, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
)
X_test_kernel_pca = kernel_pca.fit(Y_mask[:,40,0].reshape(1,-1))

#%%
kernal1=X_test_kernel_pca[:,0].reshape(Nx,Ny)
kernal2=X_test_kernel_pca[:,1].reshape(Nx,Ny)
plt.scatter(kernal1)
plt.scatter(kernal2)

#%%
#threshold = 1*np.std(Y,axis=0)
#Y_mask[abs(Y)<threshold]=0
mask1=Y_mask.reshape((Nx,Ny,Nz,Nd))

#Second threhold:

plt.subplot(211)
plt.imshow(dti._data[:,:,40,0]*mask1[:,:,40,0],cmap='gray')
plt.subplot(212)
plt.imshow(dti._data[:,:,40,0],cmap='gray')
# %%
dti.go_calc_DTI(bclickCOM=False,bCalcHA=True,bFastOLS=True,bNumba=True)
dti.imshow_diff_params()
# %%
def setThrehold(Y,thresholdcommand='default'):
    #calculate the Y diviation:
    #The input should be NX*Ny,Nz,Nrep
    if thresholdcommand== 'default':
        threshold = 1*np.std(Y,axis=0)
    elif thresholdcommand==0:
        return Y
    else:
        threshold=np.float(thresholdcommand)
    Y[abs(Y)<threshold]=0
    return Y