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
%matplotlib inline                        
from libDiffusion_V2 import diffusion  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
        diffGrad_str=[1,0,0]
    diffGradArray.append(diffGrad_str)
    diffBVal_str=ds['0x5200','0x9230'][0]['0x0018','0x9117'][0]['0x0018','0x9087']
    diffBValArray.append(diffBVal_str.value)
# %%
#Maybe I assume 1 b0, 72 b500, and it's 12 Direction * 6
#path=r'C:\Research\MRI\Ungated\CIRC_00325\MR_ep2d_diff_moco2asym_ungated_b500_TE59_32ave'

datatmp=np.transpose(data,(1,2,0,3))
bvec=np.copy(np.array(diffGradArray))
for i in range(73):
    bvec[i,:][[1,2,0]]=np.array(diffGradArray)[i,:]
dti_20 = diffusion(data=datatmp,bval=diffBValArray,bvec=np.array(diffGradArray),ID='UCSF_7020_05162023',datasets=datasets,bFilenameSorted=False)

# %% ####################################################################################
# Calculate DTI Params ##################################################################
#########################################################################################
%matplotlib qt
dti_20.CoM = [None]*dti_20.Nz
for i in range(dti_20.Nz):
    dti_20.CoM[i] = np.array((35,40))
#  Calculate DTI 
dti_20.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False,bclickCOM=False)

#%%
%matplotlib qt

import plotly.express as px
# custom imshow to show the diffusion parameters
col_shape = (dti_20.Nx,dti_20.Ny*dti_20.Nz)
md = dti_20.md #.reshape(col_shape,order='F')
fa = np.transpose(dti_20.fa,(2,0,1)) #.reshape(col_shape,order='F')
ha = dti_20.ha #.reshape(col_shape,order='F')

#fa_stack = np.dstack((fa,)*3)
#colFA = self.pvec.reshape((self.Nx,self.Ny*self.Nz,3),order='f')/np.max(self.pvec[:])*255*fa_stack
colFA = dti_20.colFA #.reshape((self.Nx,self.Ny*self.Nz,3),order='f')*255
# swap x and y if phase encode is LR for display
temp = np.copy(colFA[...,0])
colFA[...,0] = colFA[...,1]
colFA[...,1] = temp

fig1 = px.imshow(md, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01,facet_row_spacing =0.01, template='plotly_dark',
                    color_continuous_scale='hot', zmin=0, zmax=0.003)
fig2 = px.imshow(fa, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01,facet_row_spacing =0.01,template='plotly_dark',
                    color_continuous_scale='dense', zmin=0, zmax=0.5)
fig3 = px.imshow(ha, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01,facet_row_spacing =0.01,template='plotly_dark',
                    color_continuous_scale='jet', zmin=-90, zmax=90)
fig4 = px.imshow(colFA, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01, facet_row_spacing =0.01,template='plotly_dark')

fig1.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig2.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig3.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig4.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig1.layout.height = 10000
fig2.layout.height = 10000
fig3.layout.height = 10000
fig4.layout.height = 10000
#%%
#md
fig1.show()

#%%
#fa
fig2.show()
#%%
#ha
fig3.show()
#%%
#colFa
fig4.show()

# save
#%%
dti.save(filename=r'C:\Research\MRI\exvivo\UCSF_7020_05162023\UCSF_7020_05162023.diffusion')

#%%
#Save the struct with
#data
#tensor
#md
#fa
#pvec
#evec
#evals
#ha
#bval
#G   ->bvec
#bmatrix
#params
#residuals  -> replace to colHa
from numpy.core.records import fromarrays
from scipy.io import savemat
mdict={}
mdict['data']=dti._data
mdict['tensor']=dti.tensor
mdict['md']=dti.md
mdict['fa']=dti.fa
mdict['pvec']=dti.pvec
mdict['evals']=dti.eval
mdict['ha']=dti.ha
mdict['bval']=dti.bval
mdict['G']=dti.bvec
mdict['bmatrix']=dti.b_matrix
mdict['param']=dti.datasets
mdict['colFA']=dti.colFA

savemat(r'C:\Research\MRI\exvivo\UCSF_7020_05162023\UCSF_7020_05162023',mdict)


#%%
from scipy.io import loadmat
a=loadmat(r'C:\Research\MRI\exvivo\UCSF_7020_05162023\UCSF_7020_05162023')

#%%Use the traditional DTI
%matplotlib inline                        
from libDiffusion_V2 import diffusion  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 
path=r'C:\Research\MRI\MP_EPI\CIRC_00373_22737_CIRC_00373_22737\CIRC_RESEARCH CIRC Research\MR ep2d_diff_Cima_M2_asym_5slices_b500_TE59_FOVphase37.5'
dti = diffusion(data=path)

# %% ####################################################################################
# Calculate DTI Params ##################################################################
#########################################################################################
%matplotlib qt
dti.CoM = [None]*dti.Nz
for i in range(dti.Nz):
    dti.CoM[i] = np.array((35,40))
#  Calculate DTI 
dti.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False,bclickCOM=False)

#%%
%matplotlib qt

import plotly.express as px
# custom imshow to show the diffusion parameters
col_shape = (dti.Nx,dti.Ny*dti.Nz)
md = dti.md #.reshape(col_shape,order='F')
fa = dti.fa #.reshape(col_shape,order='F')
ha = dti.ha #.reshape(col_shape,order='F')

#fa_stack = np.dstack((fa,)*3)
#colFA = self.pvec.reshape((self.Nx,self.Ny*self.Nz,3),order='f')/np.max(self.pvec[:])*255*fa_stack
colFA = dti.colFA #.reshape((self.Nx,self.Ny*self.Nz,3),order='f')*255
# swap x and y if phase encode is LR for display
temp = np.copy(colFA[...,0])
colFA[...,0] = colFA[...,1]
colFA[...,1] = temp

fig1 = px.imshow(md, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01,facet_row_spacing =0.01, template='plotly_dark',
                    color_continuous_scale='hot', zmin=0, zmax=0.003)
fig2 = px.imshow(fa, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01,facet_row_spacing =0.01,template='plotly_dark',
                    color_continuous_scale='dense', zmin=0, zmax=0.5)
fig3 = px.imshow(ha, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01,facet_row_spacing =0.01,template='plotly_dark',
                    color_continuous_scale='jet', zmin=-90, zmax=90)
fig4 = px.imshow(colFA, facet_col=2, facet_col_wrap=np.min([md.shape[2],3]), 
                    facet_col_spacing=0.01, facet_row_spacing =0.01,template='plotly_dark')

fig1.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig2.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig3.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig4.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})
fig1.layout.height = 10000
fig2.layout.height = 10000
fig3.layout.height = 10000
fig4.layout.height = 10000
#md
fig1.show()

#%%
#fa
fig2.show()
#%%
#ha
fig3.show()
#%%
#colFa
fig4.show()

# save
#%%
dti.save()

#%%
#Save the struct with
#data
#tensor
#md
#fa
#pvec
#evec
#evals
#ha
#bval
#G   ->bvec
#bmatrix
#params
#residuals  -> replace to colHa
from numpy.core.records import fromarrays
from scipy.io import savemat
mdict={}
mdict['data']=dti._data
mdict['tensor']=dti.tensor
mdict['md']=dti.md
mdict['fa']=dti.fa
mdict['pvec']=dti.pvec
mdict['evals']=dti.eval
mdict['ha']=dti.ha
mdict['bval']=dti.bval
mdict['G']=dti.bvec
mdict['bmatrix']=dti.b_matrix
dti.datasets=None
#mdict['param']=dti.datasets
mdict['colFA']=dti.colFA
filename=os.path.join(dti.path,dti.ID)
savemat(filename,mdict)


#%%
from scipy.io import loadmat
a=loadmat(filename)
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