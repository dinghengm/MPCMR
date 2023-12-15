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
from libDiffusion_exvivo import diffusion  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# %%
path=r'C:\Research\MRI\exvivo\SecondRUn\UCSF_7020 UCSF_7020\CIRC_DEVELOPMENT diffusion\MR ep2d_diff_mddw30_p2_s2'
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
    xaxis = datasets[0]['0x5200','0x9230'][0]['0x0020','0x9116'][0]['0x0020','0x0037'][0:3]
    yaxis = datasets[0]['0x5200','0x9230'][0]['0x0020','0x9116'][0]['0x0020','0x0037'][3:6]
    zaxis = np.cross(xaxis,yaxis)
    try:
        diffGrad_str=ds['0x5200','0x9230'][0]['0x0018','0x9117'][0]['0x0018','0x9076'][0]['0x0018','0x9089'].value
    except:
        diffGrad_str=[1,0,0]
    diffGradArray.append(diffGrad_str)
    diffBVal_str=ds['0x5200','0x9230'][0]['0x0018','0x9117'][0]['0x0018','0x9087']
    diffBValArray.append(diffBVal_str.value)
    diffGrad = np.zeros((Nd,3))
    diffBVal = np.zeros((Nd,))
    diffBValTarget = diffBValArray
    diffGradTarget = diffGradArray
for d in range(Nd):
    dd = int(d/range(Nd).step) #int(d/Nz)
    diffBVal[dd] = diffBValTarget[d]
    diffGrad[dd,0] = np.dot(xaxis, diffGradTarget[d])
    diffGrad[dd,1] = np.dot(yaxis, diffGradTarget[d])
    diffGrad[dd,2] = np.dot(zaxis, diffGradTarget[d])
    diffGrad[dd,:] = diffGrad[dd,:]/np.linalg.norm(diffGrad[dd,:])
# %%
#Maybe I assume 1 b0, 72 b500, and it's 12 Direction * 6
#path=r'C:\Research\MRI\Ungated\CIRC_00325\MR_ep2d_diff_moco2asym_ungated_b500_TE59_32ave'
datatmp=np.zeros((Nz,Ny,Nx,Nd))
diffGradArraytmp=[]
transMatrix=[[0,0,-1],[0,1,0],[1,0,0]]
for dd in range(Nd):
    datatmp[...,dd]=np.rot90(data[:,:,:,dd],k=1,axes=(0,2))
    diffGradArraytmp.append(np.dot(transMatrix,diffGradArray[dd]))
##
#We know the initiate bvec code will swap the bvec 0 and 1
#So swap one mor time
diffGradArraytmp=np.array(diffGradArraytmp)
temp = np.copy(diffGradArraytmp[:,0])
diffGradArraytmp[:,0] = diffGradArraytmp[:,1]
diffGradArraytmp[:,1] = temp
#%%

dti_20 = diffusion(data=datatmp,bval=diffBValArray,bvec=np.array(diffGradArraytmp),ID='UCSF_7020_05162023',datasets=datasets,bFilenameSorted=False)

# %% ####################################################################################
# Calculate DTI Params ##################################################################
#########################################################################################
%matplotlib qt
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
dti_20.save(filename=r'C:\Research\MRI\exvivo\Send1\UCSF_7020_05162023.diffusion')

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
mdict['data']=dti_20._data
mdict['tensor']=dti_20.tensor
mdict['md']=dti_20.md
mdict['fa']=dti_20.fa
mdict['pvec']=dti_20.pvec
mdict['evals']=dti_20.eval
mdict['ha']=dti_20.ha
mdict['bval']=dti_20.bval
mdict['G']=dti_20.bvec
mdict['bmatrix']=dti_20.b_matrix
mdict['colFA']=dti_20.colFA

savemat(r'C:\Research\MRI\exvivo\Send1\UCSF_7020_05162023',mdict)


#%%
from scipy.io import loadmat
a=loadmat(r'C:\Research\MRI\exvivo\UCSF_7020_05162023\UCSF_7020_05162023')
#%%
for items in a:
    print(items,f'{np.shape(a[items])}')
#%%Use the traditional DTI
%matplotlib inline                        
from libDiffusion_exvivo import diffusion  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os


import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#%%
ID='UCSF_7021'
path=rf'C:\Research\MRI\exvivo\SecondRUn\UCSF_7021_Nov_15 UCSF_7021_Nov_15\CIRC_DEVELOPMENT diffusion\MR ep2d_diff_mddw30_p2_s2'
import fnmatch,pydicom
import SimpleITK as sitk
dicom_filelist = fnmatch.filter(sorted(os.listdir(path)),'*.dcm')
if dicom_filelist == []:
    dicom_filelist = fnmatch.filter(sorted(os.listdir(path)),'*.DCM')
datasets = [pydicom.dcmread(os.path.join(path, file))
                        for file in dicom_filelist]
img = datasets[0].pixel_array
Nx, Ny = img.shape
NdNz = len(datasets)
data = np.zeros((Nx,Ny,NdNz))
sliceLocsArray = []
diffGradArray = []
diffBValArray = []
xaxis = datasets[0].ImageOrientationPatient[0:3]
yaxis = datasets[0].ImageOrientationPatient[3:6]
zaxis = np.cross(xaxis,yaxis)
for ind,ds in enumerate(datasets):
    data[:,:,ind]=ds.pixel_array
    sliceLocsArray.append(float(ds.SliceLocation))
    try:
        diffGrad_str = ds[hex(int('0019',16)), hex(int('100e',16))].repval
        diffGrad_str_array = diffGrad_str.split('[')[1].split(']')[0].split(',')
        diffGradArray.append([float(temp) for temp in diffGrad_str_array]) 
    except:
        diffGrad_str=[1.0,0.0,0.0]
        diffGradArray.append(diffGrad_str) 
    diffBVal_str = ds[hex(int('0019',16)), hex(int('100c',16))].repval
    diffBValArray.append(float(diffBVal_str.split('\'')[1]))
sliceLocs = np.sort(np.unique(sliceLocsArray)) #all unique slice locations
Nz = len(sliceLocs)
diffBValTarget = np.array(diffBValArray)
diffGradTarget = np.array(diffGradArray)
Nd = int(NdNz/Nz)
data_final = data.reshape([Nx,Ny,Nz,Nd],order='F')
#%%
diffDicomHDRRange = range(0,NdNz,Nz)

# create numpy arrays and rotate diffusion gradients into image plane
diffGrad = np.zeros((Nd,3))
diffBVal = np.zeros((Nd,))
for d in diffDicomHDRRange:
    dd = int(d/diffDicomHDRRange.step) #int(d/Nz)
    diffBVal[dd] = diffBValTarget[d]
    diffGrad[dd,0] = np.dot(xaxis, diffGradTarget[d])
    diffGrad[dd,1] = np.dot(yaxis, diffGradTarget[d])
    diffGrad[dd,2] = np.dot(zaxis, diffGradTarget[d])
    diffGrad[dd,:] = diffGrad[dd,:]/np.linalg.norm(diffGrad[dd,:])


#%%
dti = diffusion(data=data_final,bval=diffBVal,bvec=diffGrad,ID=ID,datasets=datasets)

#%%
savedir=r'C:\Research\MRI\exvivo\SecondRUn'
dti=diffusion(os.path.join(savedir,f'{ID}.diffusion'))
#%%
plt.imshow(dti._data[:,:,35,0],cmap='gray')
# %% ####################################################################################
# Calculate DTI Params ##################################################################
#########################################################################################
%matplotlib qt
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
dti.save(filename=os.path.join(savedir,f'{ID}.diffusion'))

#%%
#Save SliceThickness
#PixelSpacing
#ImagePositionPatient
#ImageOrientationPatient 
pixelSpacing=datasets[0].PixelSpacing
sliceThickness=datasets[0].SliceThickness
ImagePositionPatientList=[]
pixelSpacingList=[]
sliceThicknessList=[]
imageOrientationPatientList=[]
for ds in datasets:
    ImagePositionPatientList.append(ds.ImagePositionPatient)
    pixelSpacingList.append(ds.PixelSpacing)
    sliceThicknessList.append(ds.SliceThickness)
    imageOrientationPatientList.append(datasets[0].ImageOrientationPatient)
ImagePositionPatientList_final = np.array(ImagePositionPatientList).reshape([Nz,Nd,3],order='F')
imageOrientationPatient=datasets[0].ImageOrientationPatient
pixelSpacingList_final=np.array(pixelSpacingList).reshape([Nz,Nd,2],order='F')
sliceThicknessList_final=np.array(sliceThicknessList).reshape([Nz,Nd,1],order='F')
imageOrientationPatientList_final=np.array(imageOrientationPatientList).reshape([Nz,Nd,6],order='F')
print(f'{ID}')
print('SliceThickness:',sliceThicknessList_final)
print('PixelSpacing:',pixelSpacingList_final)
print('ImageOrientationPatient:',imageOrientationPatientList_final)

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

