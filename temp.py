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
from libMapping_v12 import mapping,readFolder,decompose_LRT  # <--- this is all you need to do diffusion processing
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
##Save it as nib
for ss in range(map.Nz):
    save_nii=os.path.join(os.path.dirname(dicomPath),f'{ss}_moco.nii.gz')
    nib.save(nib.Nifti1Image(data,affine=np.eye(4)),save_nii)
# %%
