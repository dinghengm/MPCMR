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
##############################################################
##############################################################
# This library is to segment all raw images to compare
# input: single slice data or multiple slices data
#
#
#%%
import numpy as np
import os
import pydicom
import plotly.express as px
import imageio # for gif
import multiprocessing
from roipoly import RoiPoly, MultiRoi
from matplotlib import pyplot as plt #for ROI poly and croping
from imgbasics import imcrop #for croping
from tqdm.auto import tqdm # progress bar
import pickle # to save diffusion object
import pandas as pd
from skimage.transform import resize as imresize
import nibabel as nib

##########################################################################################################
# Class Library
##########################################################################################################


class segmentation:
    def __init__(self, data=None, rawdata=None,bval=None, bvec=None,CIRC_ID='',
                ID=None,valueList=[],path=None): 
        self.version = 1.3
        self.ID = ID
        self.CIRC_ID=CIRC_ID
        self.path=path

        self.bval=bval
        self.bvec=bvec
        self.valueList=valueList
        if len(data.shape) == 4:
            [Nx, Ny, Nz, Nd] = data.shape
        elif len(data.shape) == 3:
            [Nx, Ny, Nd] = data.shape
            Nz = 1
        else:
            raise Exception('data needs to be 3D [Nx,Ny,Nd] or 4D [Nx,Ny,Nz,Nd] shape')
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nd = Nd
        self.shape = data.shape
        self._raw_data = np.copy(rawdata) #original raw data is untouched just in case we need
        self._data = np.copy(data) #this is the data we will be calculating everything off
        self.mask_endo = []
        self.mask_epi = []
        self.mask_lv = []
        self.mask_septal = []
        self.mask_lateral = []
        self.CoM = []
        self.cropzone = []
        self.roi=[]
        print('Data loaded successfully')
        

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
        
    def go_crop_Auto(self,data=None,cropStartVx=32):
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

    # save the diffusion object
    def save(self, filename=None, path=None, ID=None):
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
            if ID == None:
                ID = self.ID
            if path==None:
                path=self.path
            if filename is None:
                filename=os.path.join(os.path.dirname(path),f'{ID}_p.segmetation')
            
            with open(filename, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            
            print('Saved '+ filename +' successfully!')
        except:
            print('Failed saving!!!!')


    # visualize using plotly
    def imshow_px(self, volume=None, zmin=None, zmax=None, 
                    fps=30, cmap='gray', frameHW=None):
        if volume is None:
            volume = self._data / np.max(self._data, axis=(0,1), keepdims=True)
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
  
    

    def imshow_corrected(self,volume=None,valueList=None,ID=None,vmin=None, vmax=None,cmap='gray',plot=False,path=None):
        if volume is None:
            volume = self._data
        if path is None:
            path=os.path.dirname(self.path)
        if valueList is None:
            valueList=self.valueList
        if ID is None:
            ID=ID=str('mosiac_'+self.CIRC_ID + '_' + self.ID )
        try:
            Nx,Ny,Nz,Nd=np.shape(volume)
        except:
            Nz=1
            Nx,Ny,Nd=np.shape(volume)
            print('please check if you input is 4D volume')
        
        if Nz==1:
            Nx,Ny,Nz,Nd=np.shape(volume)
            plt.style.use('dark_background')
            fig,axs=plt.subplots(1,Nd, figsize=(Nd*3.3,Nz*3.3), constrained_layout=True)
            for d in range(Nd):
                    if vmin is None:
                        vmin = np.min(volume)
                    if vmax is None:
                        vmax = np.max(volume)
                    axs[d].imshow(volume[:,:,0,d],cmap=cmap,vmin=vmin,vmax=vmax)
                    axs[d].set_title(f'{valueList[d]}',fontsize='small')
                    axs[d].axis('off')
            img_dir= os.path.join(path,f'{ID}')
            if plot:
                plt.savefig(img_dir,bbox_inches='tight')
                plt.savefig(img_dir+'.pdf',bbox_inches='tight')
        elif len(np.shape(volume))==4:
            Nx,Ny,Nz,Nd=np.shape(volume)
            plt.style.use('dark_background')
        
            fig,axs=plt.subplots(Nz,Nd, figsize=(Nd*3.3,Nz*3),constrained_layout=True)            
            for d in range(Nd):
                for z in range(Nz):
                    if vmin is None:
                        vmin = np.min(volume[:,:,z,:])
                    if vmax is None:
                        vmax = np.max(volume[:,:,z,:])

                    axs[z,d].imshow(volume[:,:,z,d],cmap=cmap,vmin=vmin,vmax=vmax)
                    if z==0:
                        axs[z,d].set_title(f'{d}\n{valueList[d]}',fontsize='small')
                    else:
                        axs[z,d].set_title(f'{valueList[d]}',fontsize='small')
                    axs[z,d].axis('off')
        #plt.show()
        #root_dir=r'C:\Research\MRI\MP_EPI\saved_ims'
            img_dir= os.path.join(path,f'{ID}')
            if plot:
                plt.savefig(img_dir, bbox_inches='tight')
                plt.savefig(img_dir+'.pdf',bbox_inches='tight')
                
        return fig,axs
# segment LV endo and epi borders
    def go_segment_LV(
        self,
        z=-1,
        d=-1,
        reject=None,
        image_type="b0",
        roi_names=['endo', 'epi'],
        brightness=0.7,
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
        if crange is None:
            crange=[np.min(data),np.max(data)*brightness]
        if d==-1:##All Nd
            constrast=range(self.Nd)
            if reject != None:
                slices = np.delete(constrast, reject)
        if z == -1: #new ROI
            self.mask_endo = np.full((data).shape, False, dtype=bool) # [None]*self.Nz
            self.mask_epi = np.full((data).shape, False, dtype=bool) #[None]*self.Nz
            self.mask_lv = np.full((data).shape, False, dtype=bool) #[None]*self.Nz
            slices = range(self.Nz)
            
        else: # modify a specific slice for ROI
            if type(z) == int:
                slices = [z]
            else:
                slices = np.copy(z)
        for d in constrast:
            for z in slices:

                if image_type == "b0":
                    image = self._data[:,:,z,d] #np.random.randint(0,255,(255,255))
                    fig = plt.figure()
                    plt.imshow(image, cmap='gray', vmax=np.max(image)*brightness,vmin=np.min(image)*brightness)
                # draw ROIs
                plt.title('Slice '+ str(d))
                fig.canvas.manager.set_window_title('Slice '+ str(d))
                multirois = MultiRoi(fig=fig, roi_names=roi_names)
                
                if np.any(np.array(roi_names) == "endo"):
                    self.mask_endo[..., z,d] = multirois.rois['endo'].get_mask(image)
                
                if np.any(np.array(roi_names) == "epi"):
                    self.mask_epi[..., z,d] = multirois.rois['epi'].get_mask(image)
                
                self.mask_lv[..., z,d] = self.mask_epi[..., z,d]^self.mask_endo[..., z,d]
                self.roi=multirois
    
    # crop function
    def _crop(self, data=None, cropzone=None):
        shape=[]
        if data is None:
            data = np.copy(self._data)
        
        if cropzone is None:
            if self.cropzone == []:
                if self.Nz == 1:
                    try:
                        Nx, Ny, Nz,Nd = data.shape
                    except:
                        Nx,Ny,Nd=data.shape
                    img_crop, cropzone = imcrop(np.sum(data, axis=2))
                    Nx, Ny = img_crop.shape
                    shape = (Nx, Ny, Nz,Nd)
                #Sometimes Nz=1 is not defined
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
                #Read the class cropzone
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
        
        Nx,Ny,Nz,Nd=np.shape(data)
        for z in tqdm(range(Nz)):
            for d in range(Nd):
                data_crop[:,:,z,d] = imcrop(data[:,:,z,d], cropzone)

        return data_crop, cropzone    

    #automatically crop the data
    def _crop_Auto(self,data=None,cropStartVx=32):
        if data is None:
            data = np.copy(self._data)
        Nx,Ny,Nz,Nd=np.shape(data)
        if Nx > Ny:
            cropWin = int(Nx/2)
            if cropStartVx is None:
                cropStartVx = cropWin
            cropData = data[cropStartVx:(cropStartVx+cropWin),...]
        else:
            if cropStartVx is None:
                cropStartVx = cropWin
            cropWin = int(Ny/2)
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
            Nx, Ny, Nz,Nd = np.shape(self._data)
            self.shape=np.shape(self._data)
            self.Nx=Nx
            self.Ny=Ny
            self.Nz=Nz
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

    def _update_data(self,data):
        '''
            This function is to update de the data to 4D shape if Nz=1;
            Also, it will change the Nx,Ny,Nz,Nd,shape in the data structure.
        '''
        if len(np.shape(data))==3:
            #This data is Nx,Ny,Nz
            Nx,Ny,Nd=np.shape(data)
            self._data=data[:,:,np.newaxis,:]
            self.Nx=Nx
            self.Ny=Ny
            self.Nz=1
            self.Nd=Nd
            self.shape=np.shape(data)
        elif len(np.shape(data))==4:
            Nx,Ny,Nz,Nd=np.shape(data)
            self._data=data
            self.Nx=Nx
            self.Ny=Ny
            self.Nz=Nz
            self.Nd=Nd
            self.shape=np.shape(data)

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
        data_temp_raw=np.delete(self._raw_data,d,axis=axis)
        list=np.delete(self.valueList,d).tolist()
        self.valueList=list
        self._raw_data=data_temp_raw
        self._data=data_temp
        self._update()
    
    #Happy New Year!!!! by DM 1/2/2024
    ##This function is to compare with l
    def bmode(self,data=None,ID=None,x=None,y=None,plot=False,path=None):
        if path==None:
            path=os.path.dirname(self.path)
        if ID is None:
            ID=self.ID
        if data==None:
            data=self.data
        Nx,Ny,Nz,Nd=np.shape(data)
        if x==None and y==None:
            y=int(np.shape(data)[0]/2)
        if x is not None:
            A2=np.zeros((Ny,Nz,Nd),dtype=np.float64)
            A2=data[x,:,:,:]
        elif y is not None:
            A2=np.zeros((Nx,Nz,Nd),dtype=np.float64)
            A2=data[:,y,:,:]
        if Nz !=1:
            fig,axs=plt.subplots(1,Nz)
            ax=axs.ravel()
            for i in range(Nz):
                ax[i].imshow(A2[...,i,:],cmap='gray')
                ax[i].set_title(f'z={i}')
                #ax[i].axis('off')
        elif Nz==1:
            A3=np.squeeze(A2)
            plt.imshow(A3,cmap='gray')
            #plt.axis('off')
        if plot==True:
            dir=os.path.join(path,ID)
            plt.savefig(dir)
