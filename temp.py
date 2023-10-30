# %%
import numpy as np
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
import re
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import nibabel as nib
try:
    from numba import njit #super fast C-like calculation
    _global_bNumba_support = True
except:
    print('does not have numba library ... slower calculations only')
    _global_bNumba_support = False
# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libMapping_v12 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import scipy.io as sio
from tqdm.auto import tqdm # progress bar

import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 300

# %%
#This is going to read the clinical raw data and compare my own function
CIRC_ID='CIRC_00373'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MR t1map_long_t1_3slice_8mm_150_gap_MOCO')
ID = dicomPath.split('\\')[-1]
data,valueList,dcmFilesList= readFolder(dicomPath)
map=mapping(data=data,CIRC_ID=CIRC_ID,ID=ID)

map.shape=np.shape(map._data)
map.go_crop()
map.shape=np.shape(map._data)
cropzone=map.cropzone
data=map._data

# %%
#Simple way: Do it outside the class
#Read the slice TI list
import nibabel as nib
Slice0TIList=[]
Slice1TIList=[]
Slice2TIList=[]

for ss in range(len(valueList)//3):
    Slice0TIList.append(valueList[0+3*ss])
    Slice1TIList.append(valueList[1+3*ss])
    Slice2TIList.append(valueList[2+3*ss])
'''
##Save it as nib
for ss in range(map.Nz):
    save_nii=os.path.join(os.path.dirname(dicomPath),f'{ss}_moco.nii.gz')
    nib.save(nib.Nifti1Image(data,affine=np.eye(4)),save_nii)

'''
# %%
dataTmp=map._data[:,:,0,:]
def ir_fit(data=None,TIlist=None,ra=500,rb=-1000,T1=600,type='WLS',error='l2',Niter=2,searchtype='grid',
            T1bound=[1,5000]):
    aEstTmps=[]
    bEstTmps=[]
    T1EstTmps=[]
    resTmps=[]
    minIndTmps=[]
    #index = np.argsort(TIlist)
    #ydata=np.squeeze(data[index])
    for ii in range(int(len(Slice0TIList)/2)):
        invertMatrix=np.concatenate((-np.ones(ii),np.ones(len(TIlist)-ii)),axis=0)
        dataTmp=data*invertMatrix.T
        minIndTmps.append(ii)
        if searchtype=='lm':
            try: 
                T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_lm(data=dataTmp,TIlist=TIlist,
                ra=ra,rb=rb,T1=T1,type=type,error=error,Niter=Niter)
                aEstTmps.append(ra_exp)
                bEstTmps.append(rb_exp)
                T1EstTmps.append(T1_exp)
                #Get the chisquare
                resTmps.append(res)
            except:
                T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(data=dataTmp,TIlist=TIlist,T1bound=T1bound)
                aEstTmps.append(ra_exp)
                bEstTmps.append(rb_exp)
                T1EstTmps.append(T1_exp)
                #Get the chisquare
                resTmps.append(res)
        elif searchtype=='grid':
            T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(data=dataTmp,TIlist=TIlist,T1bound=T1bound)
            aEstTmps.append(ra_exp)
            bEstTmps.append(rb_exp)
            T1EstTmps.append(T1_exp)
            #Get the chisquare
            resTmps.append(res)
    returnInd = np.argmin(np.array(resTmps))
    T1_final=T1EstTmps[returnInd]
    ra_final=aEstTmps[returnInd]
    rb_final=bEstTmps[returnInd]
    return T1_final,ra_final,rb_final,resTmps[returnInd],int(minIndTmps[returnInd])
def go_ir_fit(data=None,TIlist=None,ra=1,rb=-2,T1=1200,parallel=False,type='WLS',Niter=2,error='l2',searchtype='grid',T1bound=[1,5000]):

    if len(np.shape(data))==3:
        Nx,Ny,Nd=np.shape(data)
        Nz=1
        NxNy=int(Nx*Ny)
        finalMap=np.zeros((Nx,Ny))
        finalRa=np.zeros((Nx,Ny))
        finalRb=np.zeros((Nx,Ny))
        finalRes=np.zeros((Nx,Ny))
    elif len(np.shape(data))==4:
        Nx,Ny,Nz,Nd=np.shape(data)
        finalMap=np.zeros((Nx,Ny,Nz))
        finalRa=np.zeros((Nx,Ny,Nz))
        finalRb=np.zeros((Nx,Ny,Nz))
        finalRes=np.zeros((Nx,Ny,Nz))
        NxNy=int(Nx*Ny)
    #Calculate all slices
    dataTmp=np.copy(data)
    for z in tqdm(range(Nz)):
        
        for x in range(Nx):
            for y in range(Ny):
                if Nz==1:
                    finalMap[x,y],finalRa[x,y],finalRb[x,y],finalRes[x,y],_=ir_fit(dataTmp[x,y],TIlist=TIlist,ra=ra,rb=rb,T1=T1,type=type,error=error,Niter=Niter,searchtype=searchtype,
                T1bound=T1bound)
                else:
                    finalMap[x,y,z],finalRa[x,y,z],finalRb[x,y,z],finalRes[x,y,z],_=ir_fit(dataTmp[x,y,z],TIlist=TIlist,ra=ra,rb=rb,T1=T1,type=type,error=error,Niter=Niter,searchtype=searchtype,
                T1bound=T1bound)
    return finalMap,finalRa,finalRb,finalRes
#%%
finalMap,finalRa,finalRb,finalRes=go_ir_fit(data=map._data[:,:,0,:],TIlist=Slice0TIList)
#%%
# %%
#######SEARCHGIRD=grid##############
Map_gird_0,Ra_grid_0,Rb_grid_0,Res_grid_0=go_ir_fit(data=map._data[:,:,0,:],TIlist=Slice0TIList,searchtype='grid',
            T1bound=[1,5000])
#%%
Map_gird_1,Ra_grid_1,Rb_grid_1,Res_grid_1=go_ir_fit(data=map._data[:,:,1,:],TIlist=Slice1TIList,searchtype='grid',
            T1bound=[1,5000])
Map_gird_2,Ra_grid_2,Rb_grid_2,Res_grid_2=go_ir_fit(data=map._data[:,:,2,:],TIlist=Slice2TIList,searchtype='grid',
T1bound=[1,5000])
#%%
T1final_grid_0=Map_gird_0*(np.divide(Rb_grid_0,Ra_grid_0)-np.ones(np.shape(Map_gird_0)))
T1final_grid_1=Map_gird_1*(np.divide(Rb_grid_1,Ra_grid_1)-np.ones(np.shape(Map_gird_1)))
T1final_grid_2=Map_gird_2*(np.divide(Rb_grid_2,Ra_grid_2)-np.ones(np.shape(Map_gird_2)))
T1final_Grid=np.stack((Map_gird_0,Map_gird_1,Map_gird_2),axis=-1)
crange=[0,5000]
#%%
%matplotlib qt
map._map=T1final_Grid
map.crange=crange
map.cropzone=cropzone
print(map.shape)
map.go_segment_LV(image_type='map',crange=crange)
#%%
data_path=os.path.dirname(dicomPath)
T1_bssfp,_,_  = readFolder(os.path.join(data_path,r'MR t1map_long_t1_3slice_8mm_150_gap_MOCO_T1-2'))
data=T1_bssfp.squeeze()
map_C=mapping(data=np.expand_dims(data,axis=-1),ID='T1_Clinical',CIRC_ID=CIRC_ID)
cropzone=map.cropzone
map_C.cropzone=cropzone
map_C.go_crop()
map_C.shape=np.shape(map_C._data)
# %%
plt.subplot(321)
plt.imshow(Map_gird_0,cmap='magma',vmin=0,vmax=5000)
plt.subplot(322)
plt.imshow(map_C._data[:,:,0],cmap='magma',vmin=0,vmax=5000)
plt.subplot(323)
plt.imshow(Map_gird_1,cmap='magma',vmin=0,vmax=5000)
plt.subplot(324)
plt.imshow(map_C._data[:,:,1],cmap='magma',vmin=0,vmax=5000)
plt.subplot(325)
plt.imshow(Map_gird_2,cmap='magma',vmin=0,vmax=5000)
plt.subplot(326)
plt.imshow(map_C._data[:,:,2],cmap='magma',vmin=0,vmax=5000)


#%%
Map_gird_0,Ra_grid_0,Rb_grid_0,Res_grid_0=go_ir_fit(data=map._data[:,:,0,:],TIlist=Slice0TIList,searchtype='lm',type='OLS',T1=1200)
#%%
Map_gird_1,Ra_grid_1,Rb_grid_1,Res_grid_1=go_ir_fit(data=map._data[:,:,1,:],TIlist=Slice1TIList,searchtype='grid',
            T1bound=[1,5000])
Map_gird_2,Ra_grid_2,Rb_grid_2,Res_grid_2=go_ir_fit(data=map._data[:,:,2,:],TIlist=Slice2TIList,searchtype='grid',
T1bound=[1,5000])