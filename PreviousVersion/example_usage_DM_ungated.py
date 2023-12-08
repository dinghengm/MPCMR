#%%
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
import nibabel as nib
import scipy.io as sio
from sklearn import preprocessing

try:
    from numba import njit #super fast C-like calculation
    _global_bNumba_support = True
except:
    print('does not have numba library ... slower calculations only')
    _global_bNumba_support = False


#from libMapping import mapping,readFolder

from libMapping_v12 import *
%matplotlib qt

# %%
dicomPath=rf'C:\Research\MRI\Ungated\{CIRC_ID}\MR_ep2d_diff_moco2asym_ungated_b500_TE59_32ave'
ID = dicomPath.split('\\')[-1]
npy_data,valueList,dcmList= readFolder(dicomPath)
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 
CIRC_ID='CIRC_00333'
dti_ungated = images(data=npy_data,ID=ID,CIRC_ID=CIRC_ID) 
# %%
dti._data=dti._raw_data
dti_ungated._data=dti_ungated._raw_data
dti.go_crop()
dti_ungated.go_crop()
#dti.go_crop_Auto()
#dti_ungated.go_crop_Auto()
num_slice=dti.Nz 
A1=np.copy(dti._data)
A2=np.copy(dti_ungated._data)

'''
for i in range(num_slice):
    #A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:]) #original data
    A1[:,:,i,:] = A1[...,i,:]/np.max(A1[...,i,:])
    #A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:]) #Processed data
for i in range(dti_ungated.Nz ):
    A2[:,:,i,:] = A2[...,i,:]/np.max(A2[...,i,:])
'''
#Show images:
#Outliner detection:

#%%
num_slice=dti.Nz 
num_ave=12   
fig,axs=plt.subplots(2,num_ave,figsize=(7,5))
newshape0,newshape1,*rest=np.shape(dti._map)
slice_read=4
for d in range(num_ave):
    axs[0,d].imshow(np.flip(A1[:,:,3,d].T,axis=0),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,d].axis('off')
    axs[1,d].imshow(A2[:,:,6,d],cmap='gray')
    #axs[1,d].set_title(f'TE={tetime[d]}ms',fontsize=5)
    axs[1,d].axis('off')


#%%
#SVD in the ungated. project them to the maybe 3/4 principle






#%%
#Use the KMean to cluster the images
#Maybe I can use 6 bins first
#Let's see
from sklearn import preprocessing
from sklearn.cluster import KMeans

#Please do the slice 6 first!!!

read_slice=6
Nx,Ny,_,Nd=np.shape(A2)
X_train=A2[:,:,read_slice,:].reshape(Nx*Ny,Nd)
X_train_norm = preprocessing.normalize(X_train).transpose(1,0)
fits = []
score = []
for k in range(2,12):
    kmeans = KMeans(n_clusters = k, random_state = 0, n_init='auto')
    kmeans.fit(X_train_norm)
    fits.append(kmeans)
    #score.append(silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean'))

#plt.plot(K,score)
#%%
#Read kmeans
#SKI learn what something 1 and matrix
k =12
ind=-1
fig,axs=plt.subplots(2,k-1,figsize=(7,5))
for i in range(k):
    axs[0,i].imshow(fits[ind].cluster_centers_[i,:].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,i].axis('off')
    axs[1,i].imshow(X_train_norm[i,:].reshape(Nx,Ny),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[1,i].axis('off')





# %%
#The histogram of the images:
A1=np.copy(dti._data)
B1=np.copy(dti._raw_data)
for i in range(num_slice):
    #A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:]) #original data
    A1[:,:,i,:] = A1[...,i,:]/np.max(B1[...,i,:])
    B1[:,:,i,:] = B1[...,i,:]/np.max(B1[...,i,:])
    #A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:]) #Processed data
A2=np.copy(dti_ungated._data)
B2=np.copy(dti_ungated._raw_data)
for i in range(num_slice):
    #A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:]) #original data
    A2[:,:,i,:] = A2[...,i,:]/np.max(B2[...,i,:])
    B2[:,:,i,:] = B2[...,i,:]/np.max(B2[...,i,:])
    #A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:]) #Processed data

# %%
num_slice=dti.Nz 
num_ave=40 
fig,axs=plt.subplots(2,num_ave,figsize=(num_ave*3,2*3))
newshape0,newshape1,*rest=np.shape(dti._map)
slice_read=4
for d in range(num_ave):
    axs[0,d].imshow(np.flip(A1[:,:,3,d].T,axis=0),cmap='gray')
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,d].axis('off')
    axs[1,d].imshow(A2[:,:,5,d],cmap='gray')
    #axs[1,d].set_title(f'TE={tetime[d]}ms',fontsize=5)
    axs[1,d].axis('off')
# %%
fig,axs=plt.subplots(2,num_ave,figsize=(num_ave*3,2*3))
for d in range(num_ave):
    axs[0,d].hist(A1[:,:,3,d].squeeze())
    #axs[0,d].set_title(f'TI={titime[readlist[d]]}ms',fontsize=5)
    axs[0,d].axis('off')
    axs[1,d].hist(A2[:,:,3,d].squeeze())

    #axs[1,d].set_title(f'TE={tetime[d]}ms',fontsize=5)
    axs[1,d].axis('off')
# %%
len_dti=np.shape(A1)[-1]
len_dti_ungated=np.shape(A2)[-1]
ratio=[]
for i in range(len_dti):
    ratio.append(np.sum(A1[:,:,3,i])/np.sum(B1[:,:,3,i]))
ratio_ungated=[]
for i in range(len_dti_ungated):
    ratio_ungated.append(np.sum(A2[:,:,5,i])/np.sum(B2[:,:,5,i]))
# %%
import scipy
def get_p_value(data):
    #In the order of 0-1, 1-2, 1-3
    try:
        pvalues=[]
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[1])
        pvalues.append(pvalue)
        stat,pvalue=scipy.stats.ttest_ind(data[1],data[2])
        pvalues.append(pvalue)
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[2])
        pvalues.append(pvalue)
    except:
        pvalues=[]
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[1])
        pvalues.append(pvalue)
    return pvalues
#%%
print=['dti','dti_ungated']
plot_data=[ratio,ratio_ungated]
plt.figure(figsize=(4,4))
bplot1 = plt.boxplot(plot_data,
                        patch_artist=True,
                        labels=print)  # will be used to label x-ticks





#%%
#PCA in the ungated
#Try x =20 and Y=20
A2_X20=A2[20,:,:,:]
A2_Y20=A2[:,20,:,:]
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_pca=[]
y_pca=[]
y_line=np.copy(A2_Y20)
x_line=np.copy(A2_X20)

Nx,Ny,Nz,Nd=np.shape(A2)
for i in range(Nz):
    X_plot_scaled = scaler.fit_transform(A2_X20[:,i,:])
    Y_plot_scaled = scaler.fit_transform(A2_Y20[:,i,:])
    x_line[:,i,:]=X_plot_scaled
    y_line[:,i,:]=Y_plot_scaled
    xpca=PCA()
    xpca.fit(X_plot_scaled)
    ypca=PCA()
    ypca.fit(Y_plot_scaled)
    x_pca.append(xpca)
    y_pca.append(ypca)

#%%
plt.figure()
plt.subplot(2,1,2)
show=imresize(y_line[:,4,0:].T,(420,48))
plt.imshow(y_line[:,4,0:48].T)
plt.axis('off')
plt.title('y_line')
plt.subplot(2,1,1)
plt.axhline(y=20, color='r', linestyle='-',linewidth=2)
plt.imshow(A2[:,:,4,0],cmap='gray')
plt.show()
#%%
#Show the lines
#Show the components
#Can I use the #4 slice
plt.figure()
plt.subplot(2,1,1)
plt.imshow(x_line[:,4,:],cmap='gray')
plt.axis('off')
plt.title('x_line')

plt.subplot(2,1,2)
plt.imshow(y_line[:,4,:].T,cmap='gray')
plt.axis('off')
plt.title('y_line')

#%%
fig,axes=plt.subplots(2,4)

for i in range(4):
    #axes[0,i].plot(x_pca[4].components_[i][:])
    axes[1,i].plot(y_pca[4].components_[i][0:50])
    axes[0,i].plot(y_pca[4].components_[i][:])
    axes[0,i].axis('off')
    axes[1,i].axis('off')

#%%
#Please filter this with 0-1 and filter
#TR=2700
#I should try to align them first, and then I can filter it
#Now it's all about clusterring
#Data in two way
#


#%%



# %%
#Write the code to generate the CINE DTI
#Second: Please use the linking algorithm to try to link those together
#Then persuade Chris to use the b0, b10, b50 to generate the cine images
#If 12 slices, then it will be 12 Heart Beat * 30 Frames * 3 Average = 1080s (18 mins) Or Use the Dictionary
#sIMPLE KNN methods


#Thought: Compute the SSIM/MSE/Cross-Entropy/ of all images and other images
#Thought2: Compute the SSIM and 

#%%
dicomPath=dicomPath=rf'C:\Research\MRI\Ungated\{CIRC_ID}\Ungated_Cine'
_,_,dcmFilesList=readFolder(dicomPath)
import re
def get_value(input_string):
    delay_part = input_string.split("Delay")[1]
    numeric_chars=delay_part.split('\\')[0]
    # Extract numeric characters from the string
    #numeric_chars = ''.join(filter(str.isdigit, delay_part))

    return int(numeric_chars)
dti_cine,_,dcmFilesList_sorted=readFilesList(dcmFilesList,get_value)
valueList=[]
valueList=set([get_value(i) for i in dcmFilesList_sorted])
#print(valueList)
# %%
#Generate the CINE for the images:

dti_cine = images(data=dti_cine,ID='dti_cine',CIRC_ID=CIRC_ID) 
# %%
dti_cine.go_crop_Auto()
C1=np.copy(dti_cine._data)

for i in range(dti_cine.Nz ):
    #A1[:,:,i,:] = map._raw_data[...,i,:]/np.max(map._raw_data[...,i,:]) #original data
    C1[:,:,i,:] = C1[...,i,:]/np.max(C1[...,i,:])
    #A2[:,:,i,:] = map._data[...,i,:]/np.max(map._data[...,i,:]) #Processed data

#%%

for i in range(len(valueList)):
    #dataShow=np.hstack((C1[...,i,:],A2[...,i,:],A1[...,i,:]-A2[...,i,:]))
    dti_cine.createGIF(f'{CIRC_ID}_cine_dti_{i}.gif',C1[...,i,:]*255,fps=30)





#%%
#Generate the KNN AL
#In the layer 4:First Run the Low Rank together
#Then calculate the KNN

#Generate a large matrix with layer 4

data_to_regress= np.concatenate((np.expand_dims(np.flip(A1[...,4,0].T,axis=0),axis=-1),A2[...,4,:]),axis=-1)


#data_regressed=_regress(data_to_regress,100,1,2)
#%
#%%
#FOR SVD:
Nx,Ny,Nd=np.shape(data_to_regress)
data=np.reshape(data_to_regress,(Nx*Ny,Nd))
U,S,VT=np.linalg.svd(data[:,0:100],full_matrices=False)
fig = plt.figure(1, figsize = [8, 8], dpi = 300)
j=1
for r in range(10):
    Xappro=U[:,:r] * S[0:r] @ VT[:r,:]
    for show in range(10):
        plt.subplot(10,10,j)
        img=plt.imshow(Xappro[:,show].reshape(Nx,Ny),cmap='gray')
        plt.axis('off')
        #plt.title(f'Rank {r},images= {show}',fontsize=2)
        j+=1
j=1
for show in range(10):
    plt.subplot(10,10,j)
    img=plt.imshow(data[:,show].reshape(Nx,Ny),cmap='gray')
    plt.axis('off')
    plt.title(f'Original,images= {show}',fontsize=2)
    j+=1
plt.show()
#%%
#Try the delay with the same 50 to regress:

data_to_regress= np.concatenate((A1[...,4,:],A2[...,4,:]),axis=-1)
plt.subplot(211)
plt.imshow(data_to_regress[:,:,0],cmap='gray')
plt.subplot(212)
plt.imshow(data_to_regress[:,:,1],cmap='gray')
#%%
Nx,Ny,Nd=np.shape(data_to_regress)
data=np.reshape(data_to_regress,(Nx*Ny,Nd))
U,S,VT=np.linalg.svd(data[:,0:100],full_matrices=False)
fig = plt.figure(1, figsize = [8, 8], dpi = 300)
j=1
for r in range(10):
    Xappro=U[:,:r] * S[0:r] @ VT[:r,:]
    for show in range(10):
        plt.subplot(10,10,j)
        img=plt.imshow(Xappro[:,show].reshape(Nx,Ny),cmap='gray')
        plt.axis('off')
        #plt.title(f'Rank {r},images= {show}',fontsize=2)
        j+=1
j=1
for show in range(10):
    plt.subplot(10,10,j)
    img=plt.imshow(data[:,show].reshape(Nx,Ny),cmap='gray')
    plt.axis('off')
    plt.title(f'Original,images= {show}',fontsize=2)
    j+=1
plt.show()


#%%
#Use this same method on the normal DTI
#SEE how the contrast change with svd
#It will be important!!!
# %%
plt.figure(1)
plt.semilogy(np.diag(S))
plt.show()
plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.show()
# %%
#For single slice
def _regress(data,rank_img=10,rank_diff=1, rank_diff_resp=2,regressindex=0):
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

# %%
data_to_regress=A2[:,:,6,:]
Nx,Ny,Nd=np.shape(data_to_regress)
Uall,_,_,_,_,data_regressed=_regress(data_to_regress,regressindex=0)
fig,axes=plt.subplots(2,11,figsize=(20,4))
for i in range(22):
    axes[int(i//11),int(i%11)].imshow(data_regressed[:,:,i],cmap='gray')
    axes[int(i//11),int(i%11)].axis('off')
fig.suptitle('Regress First')
# %%
#%%
Uall,_,_,_,_,data_regressed=_regress(data_to_regress,regressindex=1)
fig,axes=plt.subplots(2,11,figsize=(20,4))

for i in range(22):
    axes[int(i//11),int(i%11)].imshow(data_regressed[:,:,i],cmap='gray')
    axes[int(i//11),int(i%11)].axis('off')
fig.suptitle('Regress Second')
#%%
Uall,_,_,_,_,data_regressed=_regress(data_to_regress,rank_diff_resp=10,regressindex=0)
fig,axes=plt.subplots(2,11,figsize=(20,4))
for i in range(22):
    axes[int(i//11),int(i%11)].imshow(data_regressed[:,:,i],cmap='gray')
    axes[int(i//11),int(i%11)].axis('off')
fig.suptitle('rank3,Regress 0')
#%%
Uall,_,_,_,_,data_regressed=_regress(data_to_regress,rank_diff_resp=3,regressindex=1)
fig,axes=plt.subplots(2,11,figsize=(20,4))
for i in range(22):
    axes[int(i//11),int(i%11)].imshow(data_regressed[:,:,i],cmap='gray')
    axes[int(i//11),int(i%11)].axis('off')
fig.suptitle('rank3,Regress 1')
#%%
Uall,_,_,_,_,data_regressed=_regress(data_to_regress,rank_diff_resp=3,regressindex=2)
fig,axes=plt.subplots(2,11,figsize=(20,4))
for i in range(22):
    axes[int(i//11),int(i%11)].imshow(data_regressed[:,:,i],cmap='gray')
    axes[int(i//11),int(i%11)].axis('off')
fig.suptitle('rank3,Regress 2')
# %%
