#########################################################################################
#########################################################################################
# CIRC's Diffusion Libraries
# MultiParametric Echo Planer Imaging (MP-EPI)
# Christopher Nguyen, PhD
# Cleveland Clinic
# 6/2022
# Dingheng Mai
# Case Western Reserve University, Biomedical Engineering
# Sep/2023
# 
#########################################################################################
#########################################################################################
#
# INSTALLATION PACKAGES PREREQ
# install anaconda first
#
# Before running this script:
#    * install anaconda
#    * from terminal, run conda activate base (or environment of choice)
#    * pip install -U tensorly
#    * pip install roipoly SimpleITK-SimpleElastix imgbasics ipyfilechooser pydicom plotly imageio PyQt6 nibabel
#    * conda install scikit-image
#    * conda install tqdm <--should already be there
# pip install ipyfilechooser
# pip install opencv-python
# python -m pip install statsmodels

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


##########################################################################################################
# Class Library
##########################################################################################################
class mapping:
    # method used to initalize object
    # data can be path to data or it can be numpy data
    def __init__(self, data=None, bval=None, bvec=None,CIRC_ID='',
                ID=None,valueList=[],datasets=None,
                UIPath=None,TE=None,sortBy='tval',reject=True): 
        
        '''
        Define class object and read in data
        
        Inputs:
         * data: select the dicom of the folder that holds all dicoms of diffusion data OR select a previous *.diffusion 
         * ID: if you don't set ID number it will be taken from DICOM
         * bMaynard: default (bMaynard=False) to process on own laptop, pass bMaynard=True to process on Maynard
        #1.1 update: change the slice location
        '''
        self.version = 1.2
        self.ID = ID
        self.CIRC_ID=CIRC_ID
        if data is None:
            fc = self._uigetfile(pattern=['*.dcm', '*.DCM', '*.mapping','*.diffusion'],bval=None,bvec=None,
                                 path = UIPath)
          #Development Mode
        if type(data) == str: #given a path so try to load in the data
            tmp=data
            if data.split('.')[-1] == 'mapping':
                print('Loading in CIRC mapping object')
                self._load(filename=data)
            elif data.split('.')[-1] in {'dcm' , 'DCM'}:
                print('Your data is in dcm form')
                self.path = os.path.dirname(os.path.abspath(data))
                data, bval, bvec, datasets = self.dicomread(self.path)  #Matthew fix one bug that now probabily load file if input is string
                self.__initialize_parameters(data=data,bval=bval,bvec=bvec,datasets=datasets)
            #Matthew add in read file from gz and npy 
            elif data.split('.')[-1] == 'gz':
                print('Your data is in nii.gz form')
                import nibabel as nib
                nii_img  = nib.load(data)
                nii_data = nii_img.get_fdata()
                self.__initialize_parameters(data=nii_data,bval=bval,bvec=bvec,valueList=valueList,datasets=datasets)
                print('Data loaded successfully')
                self.path = os.path.dirname(os.path.abspath(data))
            elif data.split('.')[-1] == 'npy':
                print('our data is in .npy form')
                npy_data=np.load(data,allow_pickle=True)
                self.__initialize_parameters(data=npy_data,bval=bval,bvec=bvec,valueList=valueList,datasets=datasets)
                print('Data loaded successfully')
                self.path = os.path.dirname(os.path.abspath(data))
            else:
            ##It will be the path to the folders
                pattern = r'CIRC_\d+'

                # Use re.findall() to find all matches of the pattern in the string
                matches = re.findall(pattern, tmp)
                CIRC_ID=matches[0]
                ID = tmp.split('\\')[-1]
                if self.ID==None:
                    self.ID=ID
                if self.CIRC_ID == None:
                    self.CIRC_ID=CIRC_ID
                npy_data,value_List,dcmList= readFolder(tmp,sortBy=sortBy,reject=reject)
                self.__initialize_parameters(data=npy_data,bval=bval,bvec=bvec,valueList=value_List,datasets=dcmList)
                self.path = tmp
        else:
            self.__initialize_parameters(data=data,bval=bval,bvec=bvec,valueList=valueList,datasets=datasets)
        # this is junk code needed to initialize to allow for the interactive code to work
        # TO DO: 
        # this can be avoided if I modify roipoly library with the 
        # updated plt.ion command instead of plt.show(block=True)
        '''
        fig = plt.figure()
        plt.imshow([  [0,0,0,1,1,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,0,1,0,0],
                    [0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0],
                    [0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,0,1,0,0],
                    [0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,0,1,0,0],
                    ], cmap = 'gray')
        plt.title('CLOSE ME')
        fig.canvas.manager.set_window_title('CLOSE ME')
        multiroi_named = MultiRoi(roi_names=['just close me', 'CLOSE ME NOW!!'])
        '''

    # initialize class parameters (needed to be a separate fcn for UI file browser callback)
    def __initialize_parameters(self,data,bval=[],bvec=[],path='',datasets=[],valueList=[]):
        if len(data.shape) == 4:
            [Nx, Ny, Nz, Nd] = data.shape
            self._map = np.zeros((Nx,Ny,Nz))
        elif len(data.shape) == 3:
            [Nx, Ny, Nd] = data.shape
            Nz = 1
            self._map = np.zeros((Nx,Ny))
        else:
            raise Exception('data needs to be 3D [Nx,Ny,Nd] or 4D [Nx,Ny,Nz,Nd] shape')
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nd = Nd
        self.shape = data.shape
        self._raw_data = np.copy(data) #original raw data is untouched just in case we need
        self._data = np.copy(data) #this is the data we will be calculating everything off
        self.dcm_list = datasets
        if self.CIRC_ID == None:
            self.CIRC_ID = self.dcm_list[0].PatientID
        try:
            reader = sitk.ImageFileReader()
            reader.SetFileName( self.dcm_list[0] )
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            TEtime=float(reader.GetMetaData('0018|0081'))
            TE = TEtime
        except:
            #Hard Code TE
            TE=40
        try:
            self.bval = np.concatenate((np.zeros(1),
                                    np.ones(Nd)*500)) #default b = 500
            self.bvec = np.concatenate((np.zeros([1,3]),
                                    self._getGradTable(Nd))) #default b = 500
        except:
            self.bval = bval
            self.bvec = bvec
        try:
            #if self.dcm_list[0][hex(int('0018',16)), hex(int('1312',16))].repval == "'ROW'":
            temp = np.copy(self.bvec[:,0])
            self.bvec[:,0] = self.bvec[:,1]
            self.bvec[:,1] = temp
                #Initiate the label:
        except:
            print('bvec is empty')
        try:
            if 'mp01' in self.ID.lower():
                valueList=sorted(valueList)
                valueList=[i+TE for i in valueList[::Nz]]
                self.valueList=valueList
                print('value:',valueList)
                self.crange=[0,3000]
                self.cmap='magma'
            elif 'mp02' in self.ID.lower():
                #valueList=[30,35,40,50,60,70,80,100]
                valueList=sorted(valueList)
                valueList=[i for i in valueList[::Nz]]
                print('value:',valueList)
                self.valueList=valueList
                self.crange=[0,150]
                self.cmap='viridis'
            elif 'mp03' in self.ID.lower():
                valueList=['b50x','b50y','b50z','b500x','b500y','b500z']
                print('value:',valueList)
                self.valueList=valueList
                self.crange=[0,3]
                self.cmap='hot'
        except:
            print('The data is the clinical data')
        

        
        self.mask_endo = []
        self.mask_epi = []
        self.mask_lv = []
        self.mask_septal = []
        self.mask_lateral = []
        self.CoM = []
        self.cropzone = []
        print('Data loaded successfully')

    # string returned if printed    
    def __str__(self):
        output_text = 'Diffusion object ' + str(self.version)
        output_text = output_text + '\n     subject ID = ' + str(self.ID)
        output_text = output_text + '\n     path = ' + str(self.path)
        output_text = output_text + '\n     data shape = ' + str(self.shape)
        output_text = output_text + '\n     bvalue shape = ' + str(self.bval.shape)
        output_text = output_text + '\n     bvec shape = ' + str(self.bvec.shape)
        
        return output_text

# ========================================================================================================
# "PUBLIC" CLASS FUNCTIONS ===============================================================================
# ========================================================================================================
    
    # crop class method
    def go_crop(self):
        '''
        Click top left and bottom right corners on pop-up window to crop
        '''
        print('Cropping of data:')
        self._data, self.cropzone = self._crop()
        self.Nx = self._data.shape[0]
        self.Ny = self._data.shape[1]
        self.shape = self._data.shape
        
    def go_crop_Auto(self,data=None,cropStartVx=40):
        cropData=self._crop_Auto(data=data,cropStartVx=cropStartVx)
        self._data=cropData
        self.Nx = self._data.shape[0]
        self.Ny = self._data.shape[1]
        self.shape = self._data.shape
        return 

    # resize class method
    def go_resize(self, scale=2, newshape=None):
        '''
        Resize data prior to motion correction
        Inputs: (either input scale or newshape)
            * scale: images will be resized to scale * (Nx, Ny)
            * newshape: tuple(Nx_new, Ny_new) images will be resized to (Nx_new, Ny_new)
        '''
        print('Resizing of data:')
        if newshape == None:
            newshape = scale*np.array([self.Nx, self.Ny])
        self.Nx = newshape[0]
        self.Ny = newshape[1]
        self._data = self._resize()

    # motion correct class method
    def go_moco(self, method='lrt', rank_img=10, rank_diff=1, rank_diff_resp=2, N_rbins=6):
        '''
        Perform motion correction
        Inputs:
            * method: 'lrt' (default) or 'naive'
            * rank_img: for lrt correction, default=10
            * rank_diff: for lrt correction, default=1
            * rank_diff_resp: for lrt correction, default=2 
        '''
        print('Motion correction of data: ')
        if method == 'lrt':
            print(' ... Performing low rank tensor motion correction')
            self._data_regress = np.copy(self._data / np.max(self._data, axis=(0, 1), keepdims=True))
            #define ranks
            for z in tqdm(range(self.Nz)):
                print(' ...... Slice '+str(z), end='')
                if self.Nz == 1:
                    data = self._data.reshape(self.Nx*self.Ny, self.Nd)
                else:
                    data = self._data[:,:,z,:].reshape(self.Nx*self.Ny, self.Nd)
                tensor = tl.tensor(data)
                
                ## Compute image LR representation
                # U1 = nvecs(data, rankImg)
                #U1 = tl.partial_svd(tensor, rank_img)[0]
                U1 = tl.svd_interface(tensor, n_eigenvecs=rank_img)[0]

                ## compute diffusion representation an projections
                # U2_diff = nvecs(data.T, rankBval)
                #U2_diff = tl.partial_svd(tensor, rank_diff)[2]
                U2_diff = tl.svd_interface(tensor, n_eigenvecs=rank_diff)[2]

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
                                                ).reshape((self.Nx,self.Ny,self.Nd))
                
                # register the images
                #t = time.time()
                if self.Nz == 1:
                    self._data_regress = data_diff_regress
                    self._data = self._coregister_elastix(data=data_diff_regress, 
                                                          orig_data=self._data)
                else:
                    self._data_regress[:,:,z,:] = data_diff_regress
                    self._data[:,:,z,:] = self._coregister_elastix(data=data_diff_regress, 
                                                                   orig_data=self._data[:,:,z,:])
                #print(' ......... ' + str(toc(t)))
            
        elif method == 'naive':
            print(' ... Performing naive motion correction')
            #define ranks
            for z in tqdm(range(self.Nz)):
                print(' ...... Slice '+str(z), end='')
                if self.Nz == 1:
                    data = self._data
                else:
                    data = np.squeeze(self._data[:,:,z,:])

                # register the images
                t = time.time()
                if self.Nz == 1:
                    self._data = self._coregister_elastix(data=data)
                else:
                    self._data[:,:,z,:] = self._coregister_elastix(data=data)
                #print(' ......... ' + str(toc(t)))


    # denoising class method
    def go_denoise(self, method='tv', weight=0.1, wavelt_rescale_sigma=True, lrt_rk=None):
        print('Denoising of data: ')
        if method == 'tv':
            print(' ... Performing TV denoising')
            for z in tqdm(range(self.Nz)):
                for d in range(self.Nd):
                    self._data[:,:,z,d] = denoise_tv_chambolle(self._data[:,:,z,d], weight=weight)
        elif method == 'wavelet':
            print(' ... Performing wavelet denoising')
            for z in tqdm(range(self.Nz)):
                for d in range(self.Nd):
                    self._data[:,:,z,d] = denoise_wavelet(self._data[:,:,z,d], rescale_sigma=wavelt_rescale_sigma)
        elif method == 'lrt':
            print(' ... Performing low rank tensor denoising')
            if lrt_rk == None:
                lrt_rk = [self.Nx/2, self.Ny/2, self.Nd] #only reduce in image domain
            
            # decompose into tucker core and factors
            core, factors = tucker(self._data, rank=lrt_rk, init='random', 
                                                        tol=1e-5, random_state=12345)
            self._data = tl.tucker_to_tensor((core, factors))


    #Calculate ADC method
    def go_cal_ADC(self):
        #Assume the first 3 is b50, the later 3 is b500
        print("Starting calculation of ADC")
        ADC_temp=np.zeros((self.Nx,self.Ny,self.Nz,3))
        S50=np.zeros((self.Nx,self.Ny))
        S500=np.zeros((self.Nx,self.Ny))
        for z in range(self.Nz):
            for d in range(int(self.Nd/2)):
                S50= self._data[...,z,d]
                S500= self._data[...,z,d+3]
                ADC_temp[:,:,z,d]=-1/450 * np.log(S500/S50)
        #S50=np.cbrt(S50)
        #S500=self._data[...,-1]*self._data[...,-2]*self._data[...,-3]
        #S500=np.cbrt(S500)
        ADCmap=np.zeros((self.Nx,self.Ny,self.Nz))
        ADCmap=np.mean(ADC_temp,axis=-1)
        self.ADC=ADCmap
        return ADCmap
    # calculate diffusion tensor class method
    def go_calc_DTI(self, bCalcHA=True, bFastOLS=False, bNumba=False):
        '''
        Calcualte diffusion tensor and DTI parameters (MD, FA, and HA)
        '''
        print('Calculating diffusion tensor for data: ')

        #first unwrap everything for faster calculation using Numba
        G = self.bvec
        bval = self.bval
        self.b_matrix = -1 * np.array([
                G[:,0]*G[:,0]*bval,  #    Bxx
                G[:,1]*G[:,1]*bval, #     Byy
                G[:,2]*G[:,2]*bval, #     Bzz
                G[:,0]*G[:,1]*2*bval, #   Bxy
                G[:,0]*G[:,2]*2*bval, #   Bxz
                G[:,1]*G[:,2]*2*bval, #   Byz
                np.ones(G[:,0].shape) # S0
            ]).T
        NxNyNz = self.Nx*self.Ny*self.Nz

        if bNumba and _global_bNumba_support:
            tensor, eval, evec = calctensor_numba(self._data.reshape([NxNyNz, self.Nd]), 
                                        self.b_matrix,
                                        bFastOLS=bFastOLS) 
        else:
            tensor, eval, evec = calctensor(self._data.reshape([NxNyNz, self.Nd]), 
                                        self.b_matrix,
                                        bFastOLS=bFastOLS) 
        eval[eval<0] = 1e-10 # do this after numba fcn since it was supported
        
        # wrap everything back up
        self.tensor = tensor.reshape([self.Nx,self.Ny,self.Nz,6])
        self.eval = eval.reshape([self.Nx,self.Ny,self.Nz,3])
        self.evec = evec.reshape([self.Nx,self.Ny,self.Nz,3,3]) # [Nx,Ny,Nz,xyz,p1p2p3]
        #temp = np.copy(self.evec[:,:,:,1,:]) #blue and green are transposed
        #self.evec[:,:,:,1,:] = self.evec[:,:,:,2,:] 
        #self.evec[:,:,:,2,:] = temp

        # calculate diffusion parametric maps
        self.md = np.mean(self.eval,axis=3)
        l1 = self.eval[...,0]
        l2 = self.eval[...,1]
        l3 = self.eval[...,2]

        self.fa = np.sqrt(((l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2) / ( l1**2 + l2**2 +l3**2 )
                        ) / np.sqrt(2)
        self.pvec = self.evec[...,0]
        self.colFA = np.abs(self.pvec)*np.tile(self.fa[:,:,:,np.newaxis],(1,1,1,3))

        if bCalcHA:
            try:
                self.ha = calcHA(self.pvec, self.CoM)
            except:
                self.go_define_CoM()
                self.ha = calcHA(self.pvec, self.CoM)
        else:
            self.ha = np.copy(self.fa)*0
        
        self.hat = calcHAT(self.ha, self.mask_lv)
        #except:
        #    self.ha = np.zeros(fa.shape)
        #    print('No LV mask!! Cannot run HA calculation: run *.go_segment_LV() class method first')
        

    def go_calc_DTI(self, bCalcHA=True, bFastOLS=False, bNumba=False, bclickCOM=False):
        '''
        Calcualte diffusion tensor and DTI parameters (MD, FA, and HA)
        '''
        print('Calculating diffusion tensor for data: ')

        #first unwrap everything for faster calculation using Numba
        G = self.bvec
        bval = self.bval
        self.b_matrix = -1 * np.array([
                G[:,0]*G[:,0]*bval,  #    Bxx
                G[:,1]*G[:,1]*bval, #     Byy
                G[:,2]*G[:,2]*bval, #     Bzz
                G[:,0]*G[:,1]*2*bval, #   Bxy
                G[:,0]*G[:,2]*2*bval, #   Bxz
                G[:,1]*G[:,2]*2*bval, #   Byz
                np.ones(G[:,0].shape) # S0
            ]).T
        NxNyNz = self.Nx*self.Ny*self.Nz

        if bNumba and _global_bNumba_support:
            tensor, eval, evec = calctensor_numba(self._data.reshape([NxNyNz, self.Nd]), 
                                        self.b_matrix,
                                        bFastOLS=bFastOLS) 
        else:
            tensor, eval, evec = calctensor(self._data.reshape([NxNyNz, self.Nd]), 
                                        self.b_matrix,
                                        bFastOLS=bFastOLS) 
        eval[eval<0] = 1e-10 # do this after numba fcn since it was supported
        
        # wrap everything back up
        self.tensor = tensor.reshape([self.Nx,self.Ny,self.Nz,6])
        self.eval = eval.reshape([self.Nx,self.Ny,self.Nz,3])
        self.evec = evec.reshape([self.Nx,self.Ny,self.Nz,3,3]) # [Nx,Ny,Nz,xyz,p1p2p3]
        #temp = np.copy(self.evec[:,:,:,1,:]) #blue and green are transposed
        #self.evec[:,:,:,1,:] = self.evec[:,:,:,2,:] 
        #self.evec[:,:,:,2,:] = temp

        # calculate diffusion parametric maps
        self.md = np.mean(self.eval,axis=3)
        l1 = self.eval[...,0]
        l2 = self.eval[...,1]
        l3 = self.eval[...,2]

        self.fa = np.sqrt(((l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2) / ( l1**2 + l2**2 +l3**2 )
                        ) / np.sqrt(2)
        self.pvec = self.evec[...,0]
        self.colFA = np.abs(self.pvec)*np.tile(self.fa[:,:,:,np.newaxis],(1,1,1,3))

        if bCalcHA:
            if bclickCOM:
                self.go_define_CoM()
                self.ha = calcHA(self.pvec, self.CoM)
            else:
                try:
                    self.ha = calcHA(self.pvec, self.CoM)
                except:
                    self.go_define_CoM()
                    self.ha = calcHA(self.pvec, self.CoM)
        else:
            self.ha = np.copy(self.fa)*0
        
        #except:
        #    self.ha = np.zeros(fa.shape)
        #    print('No LV mask!! Cannot run HA calculation: run *.go_segment_LV() class method first')
        

    # segment LV endo and epi borders
    def go_segment_LV(
        self,
        z=-1,
        reject=None,
        image_type="map", 
        cmap="gray", 
        dilate=True, 
        kernel=3, 
        roi_names=['endo', 'epi'],
        crange=None):
        '''
        Segment the LV
        
        Input:
            * z: slices to segment, -1 (default) to segment all
            * reject: slices to reject (will not have to segment, will not contribute to stats)
            * image_type: 
                    - "b0_avg": average of all b0 (really b=50) images
                    - "b0": first b0 image
                    - "MD": md map
                    - "HA": ha map
                    - "HA overlay": shows HA masked by current LV mask over b0_avg
            * cmap: color map, default is "gray" (may want "jet" for HA)
            * dilate: for "HA overlay", dilates current LV mask
            * kernel: dictates how much the LV mask is dilated (higher kernel = more dilation)
            * roi names: rois to generate, default ['endo', 'epi'], but may want ['endo', 'epi', 'septal', 'lateral']
'''
        # you will need to use this decorator to make this work --> %matplotlib qt
        print('Segment of the LV')
        data=self._data
        if crange==None:
            crange=[np.min(data),np.max(data)]
        if z == -1: #new ROI
            self.mask_endo = np.full((data).shape[:3], False, dtype=bool) # [None]*self.Nz
            self.mask_epi = np.full((data).shape[:3], False, dtype=bool) #[None]*self.Nz
            self.mask_lv = np.full((data).shape[:3], False, dtype=bool) #[None]*self.Nz
            self.mask_septal = np.full((data).shape[:3], False, dtype=bool) #[None]*self.Nz
            self.mask_lateral = np.full((data).shape[:3], False, dtype=bool) #[None]*self.Nz
            self.CoM = [None]*self.Nz
            slices = range(self.Nz)
        else: # modify a specific slice for ROI
            slices = [z]
        for z in slices:

            if image_type == "b0":
                image = self._data[:,:,z,0] #np.random.randint(0,255,(255,255))
                fig = plt.figure()
                plt.imshow(image, cmap=cmap, vmax=np.max(image)*0.6)
            
            elif image_type == "b0_avg":
                image = np.mean(self._data[:,:,z,np.argwhere(self.bval==50).ravel()], axis=-1)
                fig = plt.figure()
                plt.imshow(image, cmap=cmap, vmax=np.max(image)*0.6)
                
            elif image_type == "HA":
                image = self.ha[:,:,z]
                fig = plt.figure()
                plt.imshow(image, cmap=cmap)
                
            elif image_type == "MD":
                image = self.md[:,:,z]
                fig = plt.figure()
                plt.imshow(image, cmap=cmap, vmax=np.max(image)*0.6)
                
          
            elif image_type == "map":
                image = self._map[:,:,z]
                fig = plt.figure()
                plt.imshow(image, cmap='gray',vmin=crange[0],vmax=crange[1])
            # draw ROIs
            plt.title('Slice '+ str(z))
            fig.canvas.manager.set_window_title('Slice '+ str(z))
            multirois = MultiRoi(fig=fig, roi_names=roi_names)
            
            if np.any(np.array(roi_names) == "endo"):
                self.mask_endo[..., z] = multirois.rois['endo'].get_mask(image)
            
            if np.any(np.array(roi_names) == "epi"):
                self.mask_epi[..., z] = multirois.rois['epi'].get_mask(image)
            
            self.mask_lv[..., z] = self.mask_epi[..., z]^self.mask_endo[..., z]
            
            if np.any(np.array(roi_names) == "septal"):
                self.mask_septal[..., z] = multirois.rois['septal'].get_mask(image)
            
            if np.any(np.array(roi_names) == "lateral"):
                self.mask_lateral[..., z] = multirois.rois['lateral'].get_mask(image)
            
            # COM
            ind = np.where(self.mask_endo[..., z]>0)
            self.CoM[z] = np.array([np.mean(ind[0]), np.mean(ind[1])])

            plt.close()
        
    def go_define_CoM(self):
        '''
        Double click to define the COM for each slice
        '''
        print('Define CoM for quick HA estimate')
        self.CoM = [None]*self.Nz
        for z in range(self.Nz):
            image = self._data[:,:,z,0] #np.random.randint(0,255,(255,255))
            fig = plt.figure()
            plt.imshow(image, cmap='gray')
            plt.title('Slice '+str(z) + ", Double Click Center")
            fig.canvas.manager.set_window_title('Slice '+str(z))
            roi_CoM = RoiPoly(fig=fig, color='r')
            self.CoM[z] = np.array([np.mean(roi_CoM.y), np.mean(roi_CoM.x)])
            

    # save the diffusion object
    def save(self, filename=None, path=None, ID=None, bMaynard=False):
        '''
        Save diffusion object
        
        Inputs:
            * filename: full path where diffusion object will be saved (with ".diffusion")
                        if None (default), will save to path + '/' + ID + '.diffusion'
            * path: if filename=None, path defines where the .diffusion object is saved
                        if None (default), the path is the directory the data was loaded from
            * ID: if filename=None, ID defines the name of the saved .diffusion object
            * bMaynard: (boolean) saves to Maynard if True, default is False
        '''
        try:
            if path == None:
                path = self.path
            if ID == None:
                ID = self.ID
            if bMaynard:
                path = '/Volumes/Project/DTMRI/_DTMRI_CIRC/0CIRC'
            if filename is None:
                filename=os.path.join(os.path.dirname(path),f'{ID}_p.mapping')

            with open(filename, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
            print('Saved '+ filename +' successfully!')
        except:
            print('Failed saving!!!!')


    # calculate the diffusion parameters for LV
    def show_calc_stats_LV(self):
        '''
        Print gloabl LV MD and LV FA values
        '''
        print(f'Global LV MD: {np.mean(self._map[self.mask_lv]): .2f} +/- {np.std(self._map[self.mask_lv]): .2f} um^2/ms')



    def export_stats(self, filename=None, path=None, ID=None,CIRC_ID=None, bMaynard=False,crange=None): 
        '''
        Export stats to .csv file. If filename already exists, the stats data is appended to the existing file
        
        Inputs:
            * filename: full path where the csv file will be saved (ending in '.csv')
                        if None (default), will save to path + '/' + ID + '.csv'
                        if the file already exists, data will be added to the file
            * path: if filename=None, path defines where the csv file is saved
                        if None (default), the path is the directory the data was loaded from
            * ID: if filename=None, ID defines the name of the saved csv file
            * bMaynard: (boolean) saves to Maynard if True, default is False
        '''
        try:
            if path == None:
                path = self.path
            if ID == None:
                ID = self.ID
            if CIRC_ID == None:
                CIRC_ID = self.CIRC_ID
            if bMaynard:
                path = '/Volumes/Project/DTMRI/_DTMRI_CIRC/CIRC'
            if filename is None:
                filename=os.path.join(path, ID) + '.csv'
            if crange == None:
                crange=self.crange
            map=self._map
            map=map[(map>=crange[0]) & (map<=crange[1])]
            keys=['CIRC_ID','ID']
            stats=[CIRC_ID,ID]
            for z in range(self.Nz):
                slice_stats = [
                    np.mean(self._map[:,:,z][self.mask_lv[:,:,z]]),
                    np.mean(self._map[:,:,z][self.mask_septal[:,:,z]]),
                    np.mean(self._map[:,:,z][self.mask_lateral[:,:,z]]),
                    np.std(self._map[:,:,z][self.mask_lv[:,:,z]]),
                    np.std(self._map[:,:,z][self.mask_septal[:,:,z]]),
                    np.std(self._map[:,:,z][self.mask_lateral[:,:,z]])] 


                slice_keys=[str(f'Slice {z} global'),
                str(f'Slice {z} septal'),
                str(f'Slice {z} lateral'),
                str(f'Slice {z} global std'),
                str(f'Slice {z} septal std'),
                str(f'Slice {z} lateral std')]



                stats.extend(slice_stats)
                keys.extend(slice_keys)
            slice_stats = [
                    np.mean(self._map[self.mask_lv]),
                    np.mean(self._map[self.mask_septal]),
                    np.mean(self._map[self.mask_lateral]),
                    np.std(self._map[self.mask_lv]),
                    np.std(self._map[self.mask_septal]),
                    np.std(self._map[self.mask_lateral])] 


            slice_keys=[str(f'global'),
                str(f'septal'),
                str(f'lateral'),
                str(f'global std'),
                str(f'septal std'),
                str(f'lateral std')]
            
            stats.extend(slice_stats)
            keys.extend(slice_keys)


            data=dict(zip(keys,stats))
            self.dti_stats = pd.DataFrame(data, index=[0])
                
            if os.path.isfile(filename):    
                self.dti_stats.to_csv(filename, index=False, header=False, mode='a')
            else:
                self.dti_stats.to_csv(filename, index=False)
            
            print('Saved '+ filename +' successfully!')
        
        except:
            print('Failed export!!!')


    # read in dicoms
    def dicomread(self, dirpath='.', bFilenameSorted=True):
        # print('Path to the DICOM directory: {}'.format(dirpath))
        # load the data
        dicom_filelist = fnmatch.filter(sorted(os.listdir(dirpath)),'*.dcm')
        if dicom_filelist == []:
            dicom_filelist = fnmatch.filter(sorted(os.listdir(dirpath)),'*.DCM')
        datasets = [pydicom.dcmread(os.path.join(dirpath, file))
                                for file in tqdm(dicom_filelist)]
        
        i = 0
        img = datasets[0].pixel_array
        Nx, Ny = img.shape
        NdNz = len(datasets)
        data = np.zeros((Nx,Ny,NdNz))
        xaxis = datasets[0].ImageOrientationPatient[0:3]
        yaxis = datasets[0].ImageOrientationPatient[3:6]
        zaxis = np.cross(xaxis,yaxis)
        sliceLocsArray = []
        diffGradArray = []
        diffBValArray = []

        # first parse out all the slice locations, diff grad directions, and image data
        for ds in datasets:
            data[:,:,i] = ds.pixel_array
            i += 1
            sliceLocsArray.append(abs(float(ds.SliceLocation)))
            diffGrad_str = ds[hex(int('0019',16)), hex(int('100e',16))].repval
            diffGrad_str_array = diffGrad_str.split('[')[1].split(']')[0].split(',')
            diffGradArray.append([float(temp) for temp in diffGrad_str_array]) 
            #print(datasets[0][hex(int('0019',16)), hex(int('100e',16))])
            diffBVal_str = ds[hex(int('0019',16)), hex(int('100c',16))].repval
            diffBValArray.append(float(diffBVal_str.split('\'')[1]))
            #print(datasets[0][hex(int('0019',16)), hex(int('100c',16))])

        # check mosaic
        if 'MOSAIC' in datasets[0].ImageType:
            print('Detected Mosaic...')
            acqMat = np.array(datasets[0].AcquisitionMatrix)
            acqMat = acqMat[acqMat > 0]
            Nx = acqMat[0]
            Ny = acqMat[1]
            Rows = datasets[0].Rows
            Cols = datasets[0].Columns
            Nd = NdNz
            data2 = np.zeros((Nx,Ny,1000,Nd)) #larger numebr of slices than needed
            if Nx > Rows or np.mod(Rows,Nx) != 0:
                Nx = acqMat[2]
                Ny = acqMat[1]
            
            z = 0
            for x in range(Nx,Rows,Nx):
                for y in range(Ny,Cols,Ny):
                    data2[:,:,z,:] = data[(x-Nx):x,(y-Ny):y,:]
                    z = z+1
            Nz = z-1
            data_final = data2[:,:,0:Nz,:]
            diffBValTarget = diffBValArray
            diffGradTarget = diffGradArray
            diffDicomHDRRange = range(Nd)

        else:
            sliceLocs = np.sort(np.unique(sliceLocsArray)) #all unique slice locations
            Nz = len(sliceLocs)
        
            if bFilenameSorted:
                diffBValTarget = diffBValArray
                diffGradTarget = diffGradArray
                Nd = int(NdNz/Nz)
                data_final = data.reshape([Nx,Ny,Nz,Nd],order='F')
            else:
                # to avoid mismatch between reading file order -->organize slice locs and diff grad manually    
                # take first slice and find the order of b-values and gradients --> this is our new order
                # NB: Each slice could have their own order so we need to reorder each slice
                # this is SUPER slow but at least everything is in the right order
                print('...Trying exhausted search and sort of dicom')
                index = np.array(sliceLocsArray == sliceLocs[0]*np.ones(np.array(sliceLocsArray).shape))
                diffBValTarget = np.array(diffBValArray)[index]
                diffGradTarget = np.array(diffGradArray)[index]
                Nd = len(diffBValTarget)

                data_final = np.zeros((Nx,Ny,Nz,Nd))
                data_final[:,:,0,:] = data[:,:,index]

                for z in tqdm(range(Nz)):
                    if z == 0:
                        continue
                    sliceIndex = np.array(sliceLocsArray == 
                                        sliceLocs[z]*np.ones(np.array(sliceLocsArray).shape))
                    dataSlice = data[:,:,sliceIndex]
                    for d in range(Nd):
                        for dd in range(Nd):
                            if diffBValTarget[d] == np.array(diffBValArray)[sliceIndex][dd]:
                                if (diffGradTarget[d] == np.array(diffGradArray)[sliceIndex][dd]).all() :
                                    data_final[:,:,z,d] = dataSlice[:,:,dd]
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

        return data_final, diffBVal, diffGrad, datasets


    # create a gif from the data
    def createGIF(self, path=None, data=None, fps=10):
        if path is None:
            path = self.path
        
        if data is None:
            data = self._data

        if not (path.split('.')[-1] == 'gif'):
            path = path + '.gif' 
        imageio.mimsave(path, np.transpose(data,[2,0,1]), duration = 1./fps,loop=30)
    #Visualize MP:
    # visualize using plotly
    def imshow(self, volume=None, zmin=None, zmax=None, 
                    fps=30, cmap='gray', frameHW=None):
        if volume is None:
            volume = self._data
        if zmin is None:
            zmin = np.min(volume[:])
        if zmax is None:
            zmax = np.max(volume[:])
        if volume.ndim == 3:
            fig = px.imshow(volume, animation_frame=2,  
                            template='plotly_dark',
                            zmin = zmin,
                            zmax = zmax, 
                            binary_string=False, 
                            color_continuous_scale=cmap,
                            labels=dict(animation_frame="slice"))
            #fig.update(data=[{'customdata': np.dstack((volume, volume)),
            #        'hovertemplate': "(%{x},%{y}) <br> %{customdata[0]:0.3f}" }])
        else:
            if volume.ndim == 4:
                fig = px.imshow(volume, animation_frame=3, 
                                facet_col=2, 
                                facet_col_wrap=np.min([volume.shape[2],3]), 
                                facet_col_spacing=0.01,
                                binary_string=False,
                                color_continuous_scale=cmap,
                                template='plotly_dark',
                                zmin = zmin,
                                zmax = zmax, 
                                labels=dict(facet_col='slice',animation_frame="time"))

        #fig.update_layout(transition = {'duration':10}) # 1/fps*1000})
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.layout.height = 1000
        if frameHW != None:
            fig.layout.height = frameHW[0]
            fig.layout.width = frameHW[1]

        fig.update_layout(margin={'t':0,'b':0,'r':0,'l':0,'pad':0})

        # add annotation drawing
        fig.show(config={'modeBarButtonsToAdd':['drawline',
                                            'drawopenpath',
                                            'drawclosedpath',
                                            'drawcircle',
                                            'drawrect',
                                            'eraseshape'
                                        ]})

    def imshow_corrected(self,volume=None,valueList=None,ID=None,vmin=None, vmax=None,cmap='gray',plot=True):
        if volume is None:
            volume = self._data
        if valueList==None:
            valueList=self.valueList
        if ID==None:
            ID=self.ID
        Nx,Ny,Nz,Nd=np.shape(volume)
        plt.style.use('dark_background')
        fig,axs=plt.subplots(Nz,Nd, figsize=(Nz*3.3,Nd*3))
        for d in range(Nd):
            for z in range(Nz):
                if vmin is None:
                    vmin = np.min(volume[:,:,z,:])
                if vmax is None:
                    vmax = np.max(volume[:,:,z,:])
                axs[z,d].imshow(volume[:,:,z,d],cmap=cmap,vmin=vmin,vmax=vmax)
                axs[z,d].set_title(f'{valueList[d]}',fontsize=5)
                axs[z,d].axis('off')
        #plt.show()
        #root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
        img_dir= os.path.join(os.path.dirname(self.path),f'{self.CIRC_ID}_{ID}_Original')
        if plot:
            plt.savefig(img_dir)
        return fig,axs
    def imshow_map(self,volume=None,crange=None,cmap='gray',ID=None,plot=True):
        try:
            Nx,Ny,Nz=np.shape(self._map)
        except:
            raise Exception('Map doesn\'t exist')
            pass
        if volume is None:
            volume = self._map
        #Generate the map color
        if ID==None:
            ID=self.ID
        if crange==None:
            if 't2' in self.ID.lower() or 'mp02' in self.ID.lower():
                self.crange=[0,150]
                self.cmap='viridis'

            elif 't1' in self.ID.lower() or 'mp01' in self.ID.lower():
                self.crange=[0,3000]
                self.cmap='magma'
            elif 'dwi' in self.ID.lower() or 'mp03' in self.ID.lower():
                self.crange=[0,3]
                self.cmap='hot'
            crange=self.crange
            cmap=self.cmap

        num_slice=self.Nz
        figsize = (3.4*num_slice, 3)

        fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
        for sl in range(num_slice):
            axes[sl].set_axis_off()
            im = axes[sl].imshow(np.flip(volume[..., sl],axis=0), vmin=crange[0],vmax=crange[1], cmap=cmap)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
        img_dir= os.path.join(os.path.dirname(self.path),f'{self.CIRC_ID}_{ID}')
        if plot:
            plt.savefig(img_dir)
        plt.show()
    
    # visualize the overlay
    def imshow_overlay(self,volume=None,crange=None,cmap='gray',ID=None,plot=True):
        try:
            Nx,Ny,Nz=np.shape(self._map)
            sl=0
            alpha = self.mask_lv[..., sl] * 1.0
        except:
            raise Exception('Map doesn\'t exist')
            pass
        if volume is None:
            volume = self._map
        #Generate the map color
        if crange==None:
            crange=self.crange
            cmap=self.cmap
        if ID==None:
            ID=self.ID
        num_slice=self.Nz
        figsize = (num_slice*3, 3)
        fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
        for sl in range(num_slice):
            alpha = self.mask_lv[..., sl] * 1.0
            base_im = self._data[:, :, sl, 0]
            brightness = 0.8
            axes[sl].set_axis_off()
            axes[sl].imshow(base_im, cmap="gray", vmax=np.max(base_im)*brightness)
            im = axes[sl].imshow(self._map[..., sl], alpha=alpha, vmin=crange[0],vmax=crange[1], cmap=cmap)
        cbar = fig.colorbar(im, ax=axes[-1], shrink=0.95, pad=0.04, aspect=11)
        
        img_dir= os.path.join(os.path.dirname(self.path),f'{self.CIRC_ID}_{ID}_overlay')
        if plot:
            plt.savefig(img_dir)
        plt.show()  

# ========================================================================================================
# "PRIVATE" CLASS FUNCTIONS ==============================================================================
# ========================================================================================================

    # load the diffusion object
    def _load(self, filename=''):
        with open(filename, 'rb') as inp:
            importDiffusion = pickle.load(inp)
        
        try:
            # update all parameters that exist in new self imported from old diffusion object
            # this is preferred over a 'copy' since it will not overwrite new members but
            # only update members that previously are defined in older object versions
            self.__dict__.update(importDiffusion.__dict__)

            print('Successfully loaded in diffusion data!!')
        except:
            print('Failed loading in *.diffusion data!')

    # UI file browser
    def _uigetfile(self,path=None, 
            pattern=['*.jpg', '*.jpeg','*.png', '*.dat', '*.nii', '*.nii.gz', '*.mat', '*.gif'],
            callbackFunc=None,
            bFilenameSorted=True # this is in the case dicome files are not sorted properly
            ):
        
        # Change defaults and reset the dialog
        if path == None:
            path = '.'
            
        # Create and display a FileChooser widget
        fc = FileChooser(path)
        display(fc)
        
        # Print the selected path, filename, or both
        #print(fc.selected_path)
        #print(fc.selected_filename)
        #print(fc.selected)     
        
        fc.default_path = path # 
        fc.default_filename = 'test.dat'
        fc.reset()

        # Shorthand reset
        fc.reset(path=fc.default_path, filename='output.txt')

        # Change hidden files
        fc.show_hidden = True

        # Show or hide folder icons
        fc.use_dir_icons = True

        # Switch to folder-only mode
        #fc.show_only_dirs = True

        # Set multiple file filter patterns (uses https://docs.python.org/3/library/fnmatch.html)
        fc.filter_pattern = pattern

        # Change the title (use '' to hide)
        fc.title = '<b> Choose a data file (' + ''.join(pattern) +') </b>'

        # Sample callback function
        def change_title(chooser):
            #fc.title = '<b>Callback function executed</b>'
            #filename = fc.selected_path +'/'+ fc.selected_filename
            #print(filename)
            if fc.selected_filename.split('.')[-1] == 'diffusion': #load in diffusion object
                # load the diffusion object
                print('Loading in CIRC diffusion object')
                self._load(filename=fc.selected_path+'/'+fc.selected_filename)                    
            else:
                # load new data form dicom
                data, bval, bvec, datasets = self.dicomread(fc.selected_path, 
                                                            bFilenameSorted=bFilenameSorted)
                self.__initialize_parameters(data=data,bval=bval,
                                             bvec=bvec,path=fc.selected_path,
                                             datasets=datasets)
            
        # Register callback function
        if callbackFunc == None:
            fc.register_callback(change_title)
        else:
            fc.register_callback(callbackFunc)

        return fc

    # crop function
    def _crop(self, data=None, cropzone=None):
        shape=[]
        if data is None:
            data = np.copy(self._data)
        
        if cropzone is None:
            if self.cropzone == []:
                if self.Nz == 1:
                    Nx, Ny, Nd = data.shape
                    img_crop, cropzone = imcrop(np.sum(data, axis=2))
                    Nx, Ny = img_crop.shape
                    shape = (Nx, Ny, Nd)
                elif len(np.shape(data))==3:
                    Nx, Ny, Nz = data.shape
                    img_crop, cropzone = imcrop(np.sum(data, axis=2))
                    Nx, Ny = img_crop.shape
                    shape = (Nx, Ny, Nz)
                else:
                    Nx, Ny, Nz, Nd = data.shape
                    img_crop, cropzone = imcrop(np.sum(np.sum(data, axis=2), axis=2))
                    Nx, Ny = img_crop.shape
                    shape = (Nx, Ny, Nz, Nd)
            else:
                try:
                    cropzone = self.cropzone
                    temp = imcrop(data[:,:,0,0], cropzone)
                    shape = (temp.shape[0], temp.shape[1], data.shape[2], data.shape[3])
                except:
                    cropzone = self.cropzone
                    temp = imcrop(data[:,:,0], cropzone)
                    shape = (temp.shape[0], temp.shape[1], data.shape[2])
        # apply crop
        data_crop = np.zeros(shape) #use updated shape
        
        Nx,Ny,Nz,*Nd=np.shape(data)
        for z in tqdm(range(Nz)):
            for d in range(Nd):
                data_crop[:,:,z,d] = imcrop(data[:,:,z,d], cropzone)

        return data_crop, cropzone    




    #automatically crop the data
    def _crop_Auto(self,data=None,cropStartVx=40):
        if data is None:
            data = np.copy(self._data)
        Nx,Ny,Nz,Nd=np.shape(data)
        if Nx > Ny:
            cropWin = int(Nx/3)
            if cropStartVx is None:
                cropStartVx = cropWin
            cropData = data[cropStartVx:(cropStartVx+cropWin),...]
        else:
            if cropStartVx is None:
                cropStartVx = cropWin
            cropWin = int(Ny/3)
            cropData = data[:,cropStartVx:(cropStartVx+cropWin),...]
        return cropData



    def _resize(self,  data=None, newshape=None):
        if data is None:
            data = np.copy(self._data)
        if newshape is None:
            newshape = (self.Nx,self.Ny,self.Nz,self.Nd)

        new_data = np.zeros(newshape)
        for z in tqdm(range(self.Nz)):
            new_data[:,:,z,:] = imresize(data[:,:,z,:],(newshape[0],newshape[1]))
        return new_data
    
    def _update(self):
        #For single slice
        if self.Nz == 1:
            Nx, Ny, Nd = np.shape(self._data)
            self.shape=np.shape(self._data)
            self.Nx=Nx
            self.Ny=Ny
            self.Nd=Nd
        # For t1 t2 clinical maps
        elif len(np.shape(self._data))==3:
            Nx, Ny, Nz = np.shape(self._data)
            self.shape=np.shape(self._data)
            self.Nx=Nx
            self.Ny=Ny
            self.Nz=Nz
            self.Nd=0
        #Then the data is 4D
        else:
            Nx, Ny, Nz, Nd = np.shape(self._data)
            self.shape=np.shape(self._data)
            self.Nx=Nx
            self.Ny=Ny
            self.Nz=Nz
            self.Nd=Nd

    def _update_mask(self,segmented):
        '''
            This function will get the mask from another input
            *input
                Segemented: Please input .diffusion or .mapping
        
        '''
        self.CoM=segmented.CoM
        self.mask_lv=segmented.mask_lv
        self.mask_endo=segmented.mask_endo
        self.mask_epi=segmented.mask_epi
        try:
            self.mask_septal=segmented.mask_septal
            self.mask_lateral=segmented.mask_lateral
        except:
            print('Not implement septal and lateral')



    def _delete(self,d=[],axis=-1):
        '''
            This function only work on T1 mapping only
            It can delete the repetition you don't want
            *input
                d=[], the repetition don't want
                axis=-1, by default delete the repetiton
        
        '''
        #Only for T1 LIST ONLY
        data_temp=np.delete(self._data,d,axis=axis)
        list=np.delete(self.valueList,d).tolist()
        self.valueList=list
        self._data=data_temp
        self._update()
    
    def _save_nib(self,data=None,path=None,ID=None):
        '''
            This function save the data as .nii file
        
        
        '''
        if path ==None:
            path=self.path
        if ID== None:
            ID=self.ID
        if data==None:
            data=self._data
        save_nii=os.path.join(os.path.dirname(path),f'{ID}_moco.nii.gz')
        nib.save(nib.Nifti1Image(data,affine=np.eye(4)),save_nii)
        print('Saved successfully!')

    # register images with simple elastix
    def _coregister_elastix(self, data=None, orig_data=None, target_index=0, regMethod="affine", #"rigid", 
                                metric="AdvancedMattesMutualInformation", 
                                interpolator='BSplineInterpolator', 
                                ResampleInterpolator="FinalBSplineInterpolator",
                                optimizer='AdaptiveStochasticGradientDescent', 
                                NumberOfSpatialSamples= 2048,#4096, 
                                NumberOfSamplesForExactGradient= 2048, #4096,
                                NumberOfIterations=50,
                                numThreads= int(multiprocessing.cpu_count() / 1), # use 1/2 of CPUs
                                verbose=False):
        
        if data is None:
            data = np.copy(self._data)
        data_reg = np.copy(data)
        sitk.ProcessObject.SetGlobalWarningDisplay(False) #suppress annoying warnings
        elastixImageFilter = sitk.ElastixImageFilter()
        transformixImageFilter = sitk.TransformixImageFilter()
        params = sitk.GetDefaultParameterMap(regMethod,)
        params["Metric"] = (metric,)
        params["Interpolator"] = (interpolator,)
        params["Optimizer"] = (optimizer, )
        params["ResampleInterpolator"] = (ResampleInterpolator,)
        params["NumberOfSpatialSamples"] = (str(NumberOfSpatialSamples),)
        params["NumberOfSamplesForExactGradient"] = (str(NumberOfSamplesForExactGradient),)
        params["MaximumNumberOfIterations"] = (str(NumberOfIterations), )
        params["NumberOfResolutions"] = (str(4), )
        params["NewSamplesEveryIteration"] = ('true', )
        #params["ImagePyramidSchedule"] = (str([8, 8, 2, 2, 1, 1]), )
        elastixImageFilter.SetParameterMap(params)
        elastixImageFilter.SetNumberOfThreads(numThreads)
        if verbose:
            elastixImageFilter.LogToConsoleOn()
            transformixImageFilter.LogToConsoleOn()
        else:
            elastixImageFilter.LogToConsoleOff()
            transformixImageFilter.LogToConsoleOff()

        #print('... Using '+str(numThreads)+' threads ... ')
        #print(' ... Diff Meas ', end='')
        Nd = data.shape[-1]

        for d in range(Nd): #tqdm(range(Nd)):
            #if d % np.round(Nd*0.1) == 0:
            #    print(str(int(d*1./Nd*100))+'%,', end='')
            fixedImage = sitk.GetImageFromArray(data[:,:,target_index].T)
            movingImage = sitk.GetImageFromArray(data[:,:,d].T)
            elastixImageFilter.SetMovingImage( movingImage )
            elastixImageFilter.SetFixedImage( fixedImage )
            
            # run registration
            # return registration directly on input data (default)
            elastixImageFilter.Execute()
            resultImage = elastixImageFilter.GetResultImage()
            
            # get transformation params and save them as transformix filter
            # run registration on the original data using the transform from the input data
            # this is the case for LRT
            if not (orig_data is None):
                transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
                orig_data_MovingImage = sitk.GetImageFromArray(orig_data[:,:,d].T)
                transformixImageFilter.SetMovingImage(orig_data_MovingImage)
                transformixImageFilter.Execute()
                resultImage = transformixImageFilter.GetResultImage()
            
            data_reg[:,:,d] = sitk.GetArrayFromImage(resultImage).T
        
        print(' ')
        return data_reg
    
    # register images using simple ITK
    def _coregister_SITK(self, data, target_index=0, lrate=0.01, niter=1000):
        data_reg = np.copy(data)
        numberOfBins = 50
        samplingPercentage = 0.01

        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfBins)
        R.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
        R.SetMetricSamplingStrategy(R.NONE) #R.RANDOM)
        R.SetOptimizerAsGradientDescent(learningRate=lrate, numberOfIterations=niter)

        for d in range(data.shape[-1]):
            fixedImage = sitk.GetImageFromArray(data[:,:,d].T)
            movingImage = sitk.GetImageFromArray(data[:,:,target_index].T)
            tx = sitk.CenteredTransformInitializer(fixedImage, movingImage, 
                                                   sitk.AffineTransform(fixedImage.GetDimension()) 
                                                   #sitk.Similarity2DTransform()
                                                   )
            R.SetInitialTransform(tx)
            R.SetInterpolator(sitk.sitkLinear)
            outTx = R.Execute(fixedImage, movingImage)
            regImage = sitk.Resample(movingImage, fixedImage, 
                                      outTx, sitk.sitkLinear, 0.0, movingImage.GetPixelID())

            data_reg[:,:,d] = sitk.GetArrayFromImage(regImage).T
        return data_reg

    # get preset gradient tables
    def _getGradTable(self, ndir, vendor='siemens'):
        if vendor == 'siemens':
            if ndir == 3:
                return np.array([   [1.0,  0.0, 0.0],
                                    [0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                ])
            elif ndir == 6:
                return np.array([   [1.0,  0.0, 1.0],
                                    [-1.0, 0.0, 1.0],
                                    [0.0, 1.0, 1.0],
                                    [0.0, 1.0, -1.0],
                                    [1.0, 1.0, 0.0],
                                    [-1.0, 1.0, 0.0],
                                ])
            elif ndir == 10:
                return np.array([   [0.000000, 0.809017,  0.618034],
                                    [0.000000, 0.190983,  1.000000],
                                    [-0.587785, 0.809017,  0.190983],
                                    [-0.951057, 0.190983,  0.309017 ],
                                    [ -0.363271, 0.809017, -0.500000],
                                    [-0.587785, 0.190983, -0.809017],
                                    [0.363271, 0.809017, -0.500000],
                                    [0.587785, 0.190983, -0.809017],
                                    [0.587785, 0.809017,  0.190983],
                                    [0.951057, 0.190983,  0.309017],
                                ])
            elif ndir == 12:
                return np.array([   [1.000000, 0.414250, -0.414250],
                                    [1.000000, -0.414250, -0.414250],
                                    [1.000000, -0.414250, 0.414250],
                                    [1.000000, 0.414250, 0.414250],
                                    [0.414250, 0.414250, 1.000000],
                                    [0.414250, 1.000000, 0.414250],
                                    [0.414250, 1.000000, -0.414250],
                                    [0.414250, 0.414250, -1.000000],
                                    [0.414250, -0.414250, -1.000000],
                                    [0.414250, -1.000000, -0.414250],
                                    [0.414250, -1.000000, 0.414250],
                                    [0.414250, -0.414250, 1.000000],
                                ])
            elif ndir == 30:
                return np.array([   [-0.208098,  0.525514,  0.850005],
                                    [0.2023870,  0.526131,  0.851002],
                                    [0.4099560,  0.175267,  0.918257],
                                    [-0.412630,  0.742620,  0.565889],
                                    [-0.207127,  0.959492,  0.280092],
                                    [-0.872713,  0.525505,  0.064764],
                                    [-0.746815,  0.526129,  0.455449],
                                    [-0.415238,  0.175473,  0.915841],
                                    [-0.746636,  0.175268,  0.673642],
                                    [-0.665701,  0.742619, -0.217574],
                                    [-0.330391,  0.959489, -0.110450],
                                    [-0.331275,  0.525513, -0.809983],
                                    [-0.663936,  0.526130, -0.569521],
                                    [-0.999332,  0.175474, -0.111904],
                                    [-0.871398,  0.175267, -0.501922],
                                    [ 0.001214,  0.742616, -0.700356],
                                    [ 0.002949,  0.959483, -0.348370],
                                    [ 0.667975,  0.525509, -0.565356],
                                    [ 0.336490,  0.526126, -0.807431],
                                    [ 0.202383, -0.175470,  0.985002],
                                    [ 0.208094,  0.175265, -0.983848],
                                    [ 0.666452,  0.742619, -0.215262],
                                    [ 0.332212,  0.959489, -0.104850],
                                    [ 0.205064,  0.958364,  0.285421],
                                    [ 0.412630,  0.742620,  0.565889],
                                    [ 0.746093,  0.175315,  0.674232],
                                    [ 0.744110,  0.525505,  0.460568],
                                    [ 0.871894,  0.526125,  0.070507],
                                    [ 0.874264,  0.175471, -0.496841],
                                    [ 1.000000,  0.175267, -0.106112],
                                ])
            else:
                raise Exception('only supports 3, 6, 10, 12 30 ... ask chris to offer more')
        else:
            raise Exception('sorry only implemented Siemens grad table for now')



#########################################################################################
# Helper functions -- NOT a class function accessible by the object
#########################################################################################

# calc tensor with Numba
# needed to be separate because Numba doesnt understand class object type
#@njit #(parallel=True)
def calctensor(data, b_matrix, bFastOLS=False):
    NxNyNz = data.shape[0]
    S = np.zeros((data.shape[1],), dtype=np.float64)
    logS = np.zeros((data.shape[1],), dtype=np.float64)
    tensor_map = np.zeros((NxNyNz,6), dtype=np.float64)
    eval_map = np.zeros((NxNyNz,3), dtype=np.float64)
    evec_map = np.zeros((NxNyNz,3,3), dtype=np.float64)
    evec = np.zeros((3,3), dtype=np.float64)
    eval = np.zeros((3,), dtype=np.float64)
    D = np.zeros((7,), dtype=np.float64)

    A_pinv = np.linalg.pinv(b_matrix)
    for i in tqdm(range(NxNyNz)):
    #for i in range(NxNyNz):
        S = np.abs(data[i,:])
        S[S==0] = 1
        logS = np.log(S)
        if bFastOLS:
            D = np.dot(A_pinv, logS)
        else:
            # remove rcond=None if using numba
            #D = np.linalg.lstsq(b_matrix, logS, rcond=None)[0] 
            [Q,R] = np.linalg.qr(b_matrix)
            R = R[:,0:b_matrix.shape[0]]
            D = np.linalg.inv(R).dot(Q.T.dot(logS))
        D[np.isnan(D)] = 0.0
        D[np.isinf(D)] = 0.0
        #D = b_matrix \ logS; %LU decomposition
        tensor_map[i,:] = D[0:6]
        #numba only supports 'no domain change'
        # so we need to cast as complex and then strip real later
        D_mat = np.array([[D[0], D[3], D[4]],
                        [D[3], D[1], D[5]],
                        [D[4], D[5], D[2]]], dtype=np.complex128)
        eval, evec = np.linalg.eigh(D_mat) 
        eval = eval.real
        evec = evec.real
        eval[eval < 0] = 1e-16
        ind = np.argsort(eval)
        eval_map[i,:] = eval[ind[::-1]] 
        evec_map[i,:,:] = evec[:,ind[::-1]]
    return tensor_map, eval_map, evec_map    

# calc tensor with Numba
# needed to be separate because Numba doesnt understand class object type
if _global_bNumba_support:
    @njit #(parallel=True)
    def calctensor_numba(data, b_matrix, bFastOLS=False):
        NxNyNz = data.shape[0]
        S = np.zeros((data.shape[1],1), dtype=np.float64)
        logS = np.zeros((data.shape[1],1), dtype=np.float64)
        tensor_map = np.zeros((NxNyNz,6), dtype=np.float64)
        eval_map = np.zeros((NxNyNz,3), dtype=np.float64)
        evec_map = np.zeros((NxNyNz,3,3), dtype=np.float64)
        evec = np.zeros((3,3), dtype=np.float64)
        eval = np.zeros((3,), dtype=np.float64)
        D = np.zeros((7,), dtype=np.float64)

        A_pinv = np.linalg.pinv(b_matrix)
        for i in range(NxNyNz):
            if np.mod(i,int(NxNyNz*0.1)) == 0:
                print('... ' + str(int(i/NxNyNz*100)) +'%')
            S = np.abs(data[i,:])
            S[S==0] = 1
            logS = np.log(S)
            if bFastOLS:
                #D = np.dot(A_pinv, logS)
                D = np.linalg.inv(b_matrix.T.dot(b_matrix)).dot(b_matrix.T).dot(logS)
            else:
                # remove rcond=None if using numba
                #D = np.linalg.lstsq(b_matrix, logS)[0] 
                [Q,R] = np.linalg.qr(b_matrix)
                R = np.linalg.inv(R[:,0:b_matrix.shape[0]])
                D = np.dot(R,np.dot(Q.T,logS))
            D[np.isnan(D)] = 0.0
            D[np.isinf(D)] = 0.0
            #D = b_matrix \ logS; %LU decomposition
            tensor_map[i,:] = D[0:6]
            #numba only supports 'no domain change'
            # so we need to cast as complex and then strip real later
            D_mat = np.array([[D[0], D[3], D[4]],
                            [D[3], D[1], D[5]],
                            [D[4], D[5], D[2]]], dtype=np.complex128)
            eval, evec = np.linalg.eig(D_mat) 
            eval = eval.real
            evec = evec.real
            eval[eval < 0] = 1e-16
            ind = np.argsort(eval)
            eval_map[i,:] = eval[ind[::-1]] 
            evec_map[i,:,:] = evec[:,ind[::-1]]
        return tensor_map, eval_map, evec_map    

# calculate helix angle
def calcHA(pvec, CoM_stack, zvec=np.array([0.,0.,1.]) #normal axis
           ):
    
    def normVecs( vecs ):
        ix = np.linalg.norm(vecs, axis=0) > 1e-6
        vecs[:,ix] = vecs[:,ix] / np.linalg.norm(vecs[:,ix], axis=0) 
        vecs[:,~ix] = np.nan
        return vecs
    
    Nx, Ny, Nz, _ = pvec.shape
    ha_map = np.zeros((Nx,Ny,Nz))
    for z in range(Nz):
        for x in range(Nx):
            for y in range(Ny):
                CoM = CoM_stack[z]

                # define tangent plane, normal axis, and radial vector
                radvec = np.array([x, y, 0.]) - np.array([CoM[0],CoM[1], 0.]) #in plane radial vector
                radvec = radvec / np.linalg.norm(radvec)
                #zvec = normVecs(zvec).T
                #zvec = np.array([0, 0, 1]) #normal axis
                tvec = np.cross(radvec.T, zvec.T) # traverse plane                
                tvec = tvec / np.linalg.norm(tvec)

                # calculate HA
                A = np.array([tvec, zvec]).T
                #proj1 = np.linalg.pinv(np.dot(A.T, A))
                #proj2 = np.dot(A, proj1)
                #proj3 = np.dot(proj2, A.T)
                #proj = np.dot(proj3,pvec[x,y,z,:].T).T
                proj3 = A.dot(np.linalg.pinv(A.T.dot(A))).dot(A.T)
                proj  = proj3.dot(pvec[x,y,z,:].T).T
                proj = proj / np.linalg.norm(proj)
                HA = np.rad2deg(np.arccos(tvec.dot(proj))) #deg
                
                # check HA for correct orientation
                cross_check = np.cross(tvec,proj)
                dot_check = np.dot(cross_check,radvec)
                if HA > 90:
                    HA = HA - 180
                
                if dot_check > 0:
                    HA = -HA

                ha_map[x,y,z] = HA
    return ha_map

def calcHAT(ha, lv_mask, NRadialSpokes=100, reject_slice=None, method="WLS", Niter=3):
    Nx, Ny, Nz = ha.shape
    thetaArray = np.linspace(0, 2*np.pi, NRadialSpokes)
    hatMean = 0.
    NTotalSpokes = 0
    if method == "OLS":
        for z in range(Nz):
            if (not np.any(np.array(reject_slice)==z)) * (not np.all(lv_mask[:,:,z] == 0)):
                ind = np.where(lv_mask[:,:,z])
                CoM = np.array([int(np.mean(ind[0])), int(np.mean(ind[1]))])
                for theta in thetaArray:
                    spoke = []
                    for r in range(np.min([Nx,Ny])):
                        x = int(r*np.cos(theta)+CoM[0])
                        y = int(r*np.sin(theta)+CoM[1])
                        if x > (Nx-1) or x < 0 or y > (Ny-1) or y < 0:
                            continue
                        if lv_mask[x,y,z]:
                            spoke.append(ha[x,y,z])
                    
                    if len(spoke) > 2:
                        transmuralDepth = np.linspace(0,100,len(spoke)) #0 to 100% endo to epi
                        hat, _ = np.polyfit(transmuralDepth, spoke, 1)
                        hatMean += hat
                        NTotalSpokes += 1

                        
    if method == "WLS":        
        for z in range(Nz):
            if (not np.any(np.array(reject_slice)==z)) * (not np.all(lv_mask[:,:,z] == 0)):
                ind = np.where(lv_mask[:,:,z])
                CoM = np.array([int(np.mean(ind[0])), int(np.mean(ind[1]))])
                for theta in thetaArray:
                    spoke = []
                    for r in range(np.min([Nx,Ny])):
                        x = int(r*np.cos(theta)+CoM[0])
                        y = int(r*np.sin(theta)+CoM[1])
                        if x > (Nx-1) or x < 0 or y > (Ny-1) or y < 0:
                            continue
                        if lv_mask[x,y,z]:
                            spoke.append(ha[x,y,z])
                    
                    if len(spoke) > 2:
                        transmuralDepth = np.linspace(0,100,len(spoke)) #0 to 100% endo to epi
                        res_sm = sm.OLS(spoke, sm.add_constant(transmuralDepth)).fit()
                        
                        for _ in range(Niter):
                            
                            res_resid = sm.OLS([abs(resid) for resid in res_sm.resid], sm.add_constant(res_sm.fittedvalues)).fit()
                            mod_fv = res_resid.fittedvalues
                            mod_fv[mod_fv == 0] = np.min(np.delete(mod_fv, np.argwhere(mod_fv==0))) # handle division by 0
                            weights = 1 / (mod_fv**2)
                            res_sm = sm.WLS(spoke, sm.add_constant(transmuralDepth), weights=weights).fit()

                        hat = res_sm.params
                            
                        hatMean += hat[1]
                        NTotalSpokes += 1

                        
    return hatMean / NTotalSpokes



# calculate helix angle transmurality
def calcHAT_perslice(ha, lv_mask, NRadialSpokes=100, reject_slice=None, method="WLS", Niter=3):
    Nx, Ny, Nz = ha.shape
    thetaArray = np.linspace(0, 2*np.pi, NRadialSpokes)
    
    HAT_perslice = []
    
    if method == "OLS":
        
        for z in range(Nz):
            hatMean = 0.
            NTotalSpokes = 0
            
            if (not np.any(np.array(reject_slice)==z)) * (not (np.all(lv_mask[:,:,z] == 0))):
                ind = np.where(lv_mask[:,:,z])
                CoM = np.array([int(np.mean(ind[0])), int(np.mean(ind[1]))])
                for theta in thetaArray:
                    spoke = []
                    for r in range(np.min([Nx,Ny])):
                        x = int(r*np.cos(theta)+CoM[0])
                        y = int(r*np.sin(theta)+CoM[1])
                        if x > (Nx-1) or x < 0 or y > (Ny-1) or y < 0:
                            continue
                        if lv_mask[x,y,z]:
                            spoke.append(ha[x,y,z])
                    
                    if len(spoke) > 2:
                        transmuralDepth = np.linspace(0,100,len(spoke)) #0 to 100% endo to epi
                        hat, _ = np.polyfit(transmuralDepth, spoke, 1)
                        hatMean += hat
                        NTotalSpokes += 1
            
            HAT_perslice.append(hatMean / np.max([NTotalSpokes, 1]))
        
        
    if method == "WLS":
        
        for z in range(Nz):
            hatMean = 0.
            NTotalSpokes = 0
            
            if (not np.any(np.array(reject_slice)==z)) * (not (np.all(lv_mask[:,:,z] == 0))):
                ind = np.where(lv_mask[:,:,z])
                CoM = np.array([int(np.mean(ind[0])), int(np.mean(ind[1]))])
                for theta in thetaArray:
                    spoke = []
                    for r in range(np.min([Nx,Ny])):
                        x = int(r*np.cos(theta)+CoM[0])
                        y = int(r*np.sin(theta)+CoM[1])
                        if x > (Nx-1) or x < 0 or y > (Ny-1) or y < 0:
                            continue
                        if lv_mask[x,y,z]:
                            spoke.append(ha[x,y,z])
                    
                    if len(spoke) > 2:
                        transmuralDepth = np.linspace(0,100,len(spoke)) #0 to 100% endo to epi
                        res_sm = sm.OLS(spoke, sm.add_constant(transmuralDepth)).fit()
                        
                        for _ in range(Niter):
                            
                            res_resid = sm.OLS([abs(resid) for resid in res_sm.resid], sm.add_constant(res_sm.fittedvalues)).fit()
                            mod_fv = res_resid.fittedvalues
                            mod_fv[mod_fv == 0] = np.min(np.delete(mod_fv, np.argwhere(mod_fv==0))) # handle division by 0
                            weights = 1 / (mod_fv**2)
                            res_sm = sm.WLS(spoke, sm.add_constant(transmuralDepth), weights=weights).fit()

                        hat = res_sm.params
                            
                        hatMean += hat[1]
                        NTotalSpokes += 1
            
            HAT_perslice.append(hatMean / np.max([NTotalSpokes, 1]))
        
                    
    return HAT_perslice    

#TODO      
#Make a weighted least square fit
#Equation: abs(ra+rb *exp(-TI/T1))
def ir_fit(data=None,TIlist=None,ra=500,rb=-1000,T1=600,type='WLS',error='l2',Niter=1):

    def ir_recovery(tVec,T1,ra,rb):
        #Equation: abs(ra+rb *exp(-tVec(TI)/T1))
        tVec=np.array(tVec)
        #Return T1Vec,ra,rb
        return ra + rb* np.exp(-tVec/T1)
    def chisquareTest(obs,exp):
        return np.sum(((obs-exp)**2/exp))
    #Make sure the data come in increasing TI-order
    index = np.argsort(TIlist)
    ydata=np.squeeze(data[index])
    #Initialize variables:
    aEstTmps=[]
    bEstTmps=[]
    T1EstTmps=[]
    resTmps=[]
    minIndTmps=[]
    minInd=np.argmin(ydata)
    if minInd==0:
        minInd=1
    elif minInd==len(TIlist):
        minInd=len(TIlist)-1
    #Invert the data to before,at,after the min
    for ii in range(2):
        if ii==0:
            #before at the min
            minIndTmp=minInd
            invertMatrix=np.concatenate((-np.ones(minIndTmp),np.ones(len(TIlist)-minIndTmp)),axis=0)
            dataTmp=ydata*invertMatrix.T
            minIndTmps.append(minIndTmp)
        if ii==1:
            #at the min
            minIndTmp=minInd+1
            invertMatrix=np.concatenate((-np.ones(minIndTmp),np.ones(len(TIlist)-minIndTmp)),axis=0)
            dataTmp=ydata*invertMatrix.T
            minIndTmps.append(minIndTmp)
        if ii==2:
            #after the min
            minIndTmp=minInd+2
            invertMatrix=np.concatenate((-np.ones(minIndTmp),np.ones(len(TIlist)-minIndTmp)),axis=0)
            dataTmp=ydata*invertMatrix.T
            minIndTmps.append(minIndTmp)
        params_OLS,params_covariance = curve_fit(ir_recovery,TIlist,dataTmp,method='lm',p0=[T1,ra,rb],maxfev=5000)
        T1_exp,ra_exp,rb_exp=params_OLS
        ydata_exp_OLS=ir_recovery(TIlist,T1_exp,ra_exp,rb_exp)
        if type=='WLS':
            for _ in range(Niter):
                ydata_exp=ir_recovery(TIlist,T1_exp,ra_exp,rb_exp)
                if error=='l1':
                    simga_square=abs(ydata_exp-dataTmp)
                elif error=='l2':
                    simga_square=abs(ydata_exp-dataTmp)**2
                weights = 1 / (simga_square)
                params_WLS,params_covariance = curve_fit(ir_recovery,TIlist,dataTmp,method='lm',p0=[T1,ra,rb],maxfev=5000,sigma=weights,absolute_sigma=True)
                #print(weights)
                T1_exp,ra_exp,rb_exp=params_WLS
            ###If the error is higher than OLS use OLS
            ydata_exp=ir_recovery(TIlist,T1_exp,ra_exp,rb_exp)
            #if chisquareTest(dataTmp,ydata_exp_OLS)<chisquareTest(dataTmp,ydata_exp):
            #    T1_exp,ra_exp,rb_exp=params_OLS
        elif type=='OLS':
            T1_exp,ra_exp,rb_exp=params_OLS
            ydata_exp=ir_recovery(TIlist,T1_exp,ra_exp,rb_exp)
        aEstTmps.append(ra_exp)
        bEstTmps.append(rb_exp)
        T1EstTmps.append(T1_exp)
        #Get the chisquare
        resTmps.append(chisquareTest(dataTmp,ydata_exp))

    returnInd = np.argmin(np.array(resTmps))
    T1_final=T1EstTmps[returnInd]
    ra_final=aEstTmps[returnInd]
    rb_final=bEstTmps[returnInd]
    #ydata_exp=ir_recovery(TIlist,T1,ra,rb)
    return T1_final,ra_final,rb_final,resTmps,int(minIndTmps[returnInd])
def go_ir_fit(data=None,TIlist=None,ra=500,rb=-1000,T1=600,parallel=False):
    #Parallel is not able to use not
    from multiprocessing import Pool
    from functools import partial
    #Go through for all slice:
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
    for z in range(Nz):
        if parallel:
            if Nz ==1:
                dataTmp=np.reshape(dataTmp,(NxNy))
                finalMap=np.reshape(finalMap,(NxNy))
            else:
                dataTmp=np.reshape(dataTmp,(NxNy,Nz,Nd))
                tempMap=np.reshape(finalMap,(NxNy,Nz))
                tempRa=np.reshape(finalRa,(NxNy,Nz))
                tempRb=np.reshape(finalRb,(NxNy,Nz))
                tempRes=np.reshape(finalRes,(NxNy,Nz))

            with Pool(6) as pool:
                temp=pool.map(partial(ir_fit,TIlist=TIlist,ra=ra,rb=-rb,T1=T1), [dataTmp[i,z,:] for i in range(NxNy)])

            finalMap[:,:,z]=np.reshape(np.array(temp[0]),(Nx,Ny))
            finalRa[:,:,z]=np.reshape(np.array(temp[1]),(Nx,Ny))
            finalRb[:,:,z]=np.reshape(np.array(temp[2]),(Nx,Ny))
            finalRes[:,:,z]=np.reshape(np.array(temp[3]),(Nx,Ny))
        else:
            for x in range(Nx):
                for y in range(Ny):
                    finalMap[x,y,z],finalRa[x,y,z],finalRb[x,y,z],_,_=ir_fit(dataTmp[x,y,z],TIlist=TIlist,ra=ra,rb=rb,T1=T1)
    return finalMap,finalRa,finalRb,finalRes

#########################################################################################
# DEPRECATED BUT USEFUL FUCNTIONS NOT PART OF DIFFUSION CLASS
#########################################################################################

# tensor decomposition LRT function which is generalized and flexible (may need to use ungated)
def decompose_LRT(data, rank_img=10, rank_diff=1, rank_diff_resp=2):
    Nx, Ny, Nz, Nd = data.shape
    data_regress = np.copy(data / np.max(data, axis=(0, 1), keepdims=True))
    data_diff_regress=np.copy(data)
    for z in tqdm(range(Nz)):
        print(' ...... Slice '+str(z), end='')
        if Nz == 1:
            data = data_regress.reshape(Nx*Ny, Nd)
        else:
            data = data_regress[:,:,z,:].reshape(Nx*Ny, Nd)
        tensor = tl.tensor(data)
        U1 = tl.svd_interface(tensor, n_eigenvecs=rank_img)[0]

        ## compute diffusion representation an projections
        # U2_diff = nvecs(data.T, rankBval)
        #U2_diff = tl.partial_svd(tensor, rank_diff)[2]
        U2_diff = tl.svd_interface(tensor, n_eigenvecs=rank_diff)[2]

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
        data_diff_regress_temp = tl.to_numpy(
                                tl.tucker_to_tensor( (G_diff_resp, (U1, U2_diff_regress)) )
                                        ).reshape((Nx,Ny,Nd))
        data_diff_regress[:,:,z,:]=data_diff_regress_temp
    return data_diff_regress

# truncate tensor based on rank helper method for LRT moco
def truncTensorDim(tensor, truncRk=(1,1)):
    U = list(range(len(truncRk)))
    for dd in range(len(truncRk)):
        # get n-mode matrix
        nMode = tl.unfold(tensor, mode=dd)

        # compute truncated partial SVD
        U[dd],s,v = tl.svd_interface(nMode.dot(nMode.T), n_eigenvecs=truncRk[dd])

    # estimate tensor core
    G = tl.tenalg.multi_mode_dot( tensor, [np.linalg.pinv(U[dd]) for dd in range(len(U))] )

    # regenerate tensor
    tensor_trunc = tl.tucker_tensor.tucker_to_tensor( (G, U) )

    return tensor_trunc, G, U
    
# this is for visualizing in 3D using a built in plotly functions (super slow)
def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

def imshow_old(volume):
    volume = volume / np.max(volume[:])
    Nx, Ny, Nz = volume.shape

    fig = go.Figure(
        frames=[go.Frame(
                        data=go.Surface(
                                        z=(Nz/10 - k * 0.1) * np.ones((Nx, Ny)),
                                        surfacecolor=np.flipud(volume[:,:,Nz - 1 - k]),
                                        cmin=0, cmax=1),
                        name=str(k) # you need to name the frame for the animation to behave properly
                        )
                for k in range(Nz)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=Nz/10 * np.ones((Nx, Ny)),
        surfacecolor=np.flipud(volume[:,:, 0]),
        colorscale='Gray',
        cmin=0, cmax=1,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=600,
            height=600,
            scene=dict(
                        zaxis=dict(range=[-0.1, 6.8], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders
    )

    fig.show()



#Read the folder, and generate a volume with Nx,Ny,Nz,Nd
#Return Volume, valueList
#Matthew Modification Sep 8th: Use the 
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

#Match the file and 
def get_value(input_string):
    pattern = r"(?i)(Ti|TE)[_]?(\d+)[ms]?"
    match = re.search(pattern, input_string)
    if match:
        return int(match.group(2))
    else:
        # Return a very large value if 'Ti' or 'ti' is not found
        return 'inf'
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
