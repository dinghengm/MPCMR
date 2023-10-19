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
from libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v12 import readFolder,decompose_LRT
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
CIRC_ID='CIRC_00373'
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI\MR000000.dcm')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP03 = mapping(data=dicomPath,ID=ID,CIRC_ID=CIRC_ID)
# %%
def get_value(filePath):
    reader = sitk.ImageFileReader()
    reader.SetFileName( filePath )
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    seriesNumber=reader.GetMetaData('0018|0082') 
    seriesNumber = int(seriesNumber)
    return seriesNumber
#Change Read_Sriers to readInverstion
def read_seriers(filePath):
    reader = sitk.ImageFileReader()
    reader.SetFileName( filePath )
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    seriesNumber=reader.GetMetaData('0020|0011') 
    seriesNumber = int(seriesNumber)
    return seriesNumber

def read_trigger(filePath,reject=False,default=327,sigma=100):
    reader = sitk.ImageFileReader()
    reader.SetFileName( filePath )
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    try:
        triggerTime=float(reader.GetMetaData('0018|1060'))
        nominalTime=float(reader.GetMetaData('0018|1062'))
    except:
        return 0
    readList=[float(default+i*nominalTime) for i in range(-5,5,1)]
    if reject:
        if min([abs(triggerTime - t)  for t in readList]) >sigma:
            return False
        else:
            return triggerTime
    return triggerTime

def readFolder(dicomPath,sortBy='tval',reject=False,default=327,sortSlice=True):
    triggerList=[]
    seriesIDList = []
    seriesFolderList = []
    dcmFilesList=[]
    valueList=[]
    datasets=[]
    seriesNumberList=[]
    for dirpath,dirs,files in  os.walk(dicomPath):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('dcm') or path.endswith('DCM'):
                if reject:
                    if read_trigger(path,reject=reject,default=default)==False:
                        continue
                    else:
                        triggerList.append(read_trigger(path))
                        dcmFilesList.append(path)
                else:
                    triggerList.append(read_trigger(path))
                    dcmFilesList.append(path)
                try:
                    seriesNumberList.append(read_seriers(path))
                    valueList.append(get_value(path))
                except:
                    print('Something Wrong with read Trigger')
    if sortBy=='seriesNumber':
        try:
            dcmFilesList=sorted(dcmFilesList,key=read_seriers)
            print(sorted(seriesNumberList))
        except:
            print('sortBy seriers number not working, try sortBy=tval')
    elif sortBy=='tval':
        try:            
            valueList=sorted(list(valueList))
            dcmFilesList=sorted(dcmFilesList,key=get_value)
        except:

            print('DWI is included, the output is not sorted')

    datasets = [pydicom.dcmread(path)
                                    for path in tqdm(dcmFilesList)]
    #print(dcmFilesList)
    sliceLocsArray=[]
    
    img = datasets[0].pixel_array
    Nx, Ny = img.shape
    NdNz = len(datasets)
    data = np.zeros((Nx,Ny,NdNz))
    print(data.shape)
    for ds in datasets:
                sliceLocsArray.append(abs(float(ds.SliceLocation)))
    sliceLocs = np.sort(np.unique(sliceLocsArray)) #all unique slice locations
    Nz = len(sliceLocs)
    Nd=int(NdNz/Nz)
    print(sliceLocs)
    data_final = data.reshape([Nx,Ny,Nz,Nd],order='F')
    j_dict={}
    if sortSlice:
        for i in range(Nz):
            j_dict[str(i)]=0
        for ds in datasets:
            i=list(sliceLocs).index(abs(float(ds.SliceLocation)))
            data_final[:,:,i,j_dict[str(i)]] = ds.pixel_array
            j_dict[str(i)]+=1
        Nd = int(NdNz/Nz)
        data_final = data.reshape([Nx,Ny,Nz,Nd],order='F')
    print(triggerList)
    return data_final,valueList,dcmFilesList

# %%
CIRC_ID='CIRC_00373'
dicomPath=os.path.join(defaultPath,F'{CIRC_ID}_22737_{CIRC_ID}_22737\t1map_long_t1_3slice_8mm_150_gap_MOCO')
ID = os.path.dirname(dicomPath).split('\\')[-1]
dcmFilesList=[]
valueList=[]
datasets=[]
seriesNumberList=[]
for dirpath,dirs,files in  os.walk(dicomPath):
    for x in files:
        path=os.path.join(dirpath,x)

        seriesNumberList.appe6nd(read_seriers(path))
        valueList.append(get_value(path))
        dcmFilesList.append(path)
#MPT1_moco,value_list,dcmlist = readFolder(dicomPath=dicomPath,sortBy='tval',reject=False,sortSlice=False)

# %%
