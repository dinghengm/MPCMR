#%%

%matplotlib inline     
import sys
sys.path.append('MPEPI')
                  
from libDiffusion_exvivo import diffusion  # <--- this is all you need to do diffusion processing

import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import h5py
from scipy.io import loadmat
image_default_root=r'C:\Research\MRI\exvivo'


import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# %%
UCSF_ID_List=['6952','6953','6954']
UCSF_ID=UCSF_ID_List[1]
dataPath=os.path.join(image_default_root,rf'{UCSF_ID}.mat')
data=loadmat(dataPath)
#data=h5py.File(os.path.join(image_default_root,rf'{UCSF_ID}.mat'),'r')
#Create the GIF and see the images:

# %%
print(UCSF_ID)

for items in data:
    print(items,f'{np.shape(data[items])}')

# %%
dti = diffusion(data=data['data_jd'],ID=f'UCSF_{UCSF_ID}')
#%%
Nd=np.shape(data['data_jd'])[-1]
dicomdata=data['data_jd']
dti._data=data['data_jd']
dti._raw_data=data['data_jd']
diffGrad=np.zeros((Nd,3))
diffBVal = np.zeros((Nd,))
diffGrad[0,:]=[1.0,0.0,0.0]
diffGrad[1::,:]=dti._getGradTable(ndir=12)
bvals=[500]*12
diffBVal[1::]=bvals
diffBVal[0]=50
# %%
dti = diffusion(data=dicomdata,bval=diffBVal,bvec=np.array(diffGrad),ID='UCSF_{UCSF_ID}',bFilenameSorted=False)

# %%
plt.imshow(dti._data[:,:,90,0],cmap='gray')
# %% ####################################################################################
# Calculate DTI Params ##################################################################
#########################################################################################
%matplotlib qt
#  Calculate DTI 
dti.go_calc_DTI(bCalcHA=True,bFastOLS=True,bNumba=False,bclickCOM=False)
#%%
import nibabel as nib
savedir=r'C:\Research\MRI\exvivo\ThirdRun'
save_nii=os.path.join(savedir,f'{UCSF_ID}.nii.gz')
nib.save(nib.Nifti1Image(dicomdata,affine=np.eye(4)),save_nii)
print('Saved successfully!')
#%%
for map in ['md','fa','ha','G','bmatrix']:
    dataTmp=data['dti_new'][map][0][0]
    savedir=r'C:\Research\MRI\exvivo\ThirdRun'
    save_nii=os.path.join(savedir,f'{UCSF_ID}_{map}.nii.gz')
    nib.save(nib.Nifti1Image(dataTmp,affine=np.eye(4)),save_nii)

# %%
dataTmp=data['dti_new']['fa'][0][0]
save_nii=os.path.join(savedir,f'{UCSF_ID}_fa.nii.gz')
nii_img  = nib.load(save_nii)
nii_data = nii_img.get_fdata()
print('dataTmp',np.shape(dataTmp))
print('nii_data',np.shape(nii_data))
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
savedir=r'C:\Research\MRI\exvivo\ThirdRun'
#%%
dti.save(filename=os.path.join(savedir,f'{UCSF_ID}.diffusion'))
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
#mdict['param']=dti.datasets
mdict['colFA']=colFA
mdict['PixelSpacing']=pixelSpacingList_final
mdict['SliceThickness']=sliceThicknessList_final
mdict['ImagePositionPatient']=ImagePositionPatientList_final
mdict['ImageOrientationPatient']=imageOrientationPatientList_final

savedir=r'C:\Research\MRI\exvivo\ThirdRun'
filename=os.path.join(savedir,f'{ID}.mat')
savemat(filename,mdict)


#%%

from scipy.io import loadmat
filename=os.path.join(savedir,f'{ID}.mat')
a=loadmat(filename)
print(f'{ID}')
for items in a:
    print(items,f'{np.shape(a[items])}')
# %%

