# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from MPEPI.libMapping_v12 import mapping,readFolder,decompose_LRT,go_ir_fit  # <--- this is all you need to do diffusion processing
from MPEPI.libMapping_v12 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams.update({'axes.titlesize': 'small'})
from t1_fitter import T1_fitter,go_fit_T1
#%%
plot=False

#%%
CIRC_ID='CIRC_00382'
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP03 = mapping(data=fr'{dicomPath}_p.mapping')
cropzone=MP03.cropzone
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_T1')
MP01 = mapping(data=fr'{dicomPath}_p.mapping')
data_8000,_,_=readFolder(dicomPath=rf'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\MR ep2d_MP01_TE_40_bright')
MP01_0 = mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_Slice0_p.mapping')
MP01_1 = mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_Slice1_p.mapping')
MP01_2 = mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\MP01_Slice2_p.mapping')
MP02 = mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2_p.mapping')
T1_MOLLI_FB=mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\T1_MOLLI_FB_p.mapping')
T1_MOLLI=mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\T1_MOLLI_p.mapping')
T2_FLASH_FB=mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\T2_FLASH_FB_p.mapping')
T2_FLASH=mapping(data=fr'C:\Research\MRI\MP_EPI\{CIRC_ID}_22737_{CIRC_ID}_22737\T2_FLASH_p.mapping')
# %%
plt.rcParams.update({'axes.titlesize': 3})
MP01_2.imshow_corrected(volume=MP01_2._data[:,:,:,[0,3,4,5,-2,-1]],valueList=['70ms','280ms','400ms','800ms','2180ms','8000ms'],ID=f'MP01_Slice2_SHOWN##',plot=True)
data2=MP02._data[:,:,2,[0,1,2,3,5,6]]
MP01_2.imshow_corrected(volume=data2[:,:,np.newaxis,:],valueList=['30ms','40ms','50ms','60ms','80ms','100ms'],ID=f'MP02_Slice2_SHOWN##',plot=True)
data3=MP03._data[:,:,2,:]
MP01_2.imshow_corrected(volume=data3[:,:,np.newaxis,:],valueList=['b50x','b50y','b50z','b500x','b500y','b500z'],ID=f'MP03_Slice2_SHOWN##',plot=True)
# %%
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
dirpath=os.path.dirname(dicomPath)
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
moco(MP03._data,MP03)
#%%
cropzone=MP03.cropzone
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2_copy')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP02.cropzone=cropzone

MP02.go_crop()
MP02.go_resize(scale=2)
moco(MP02._data,MP02)


# %%
MP02.go_moco(method='naive')
MP02.imshow_corrected(plot=True)
# %%
MP01 = mapping(data=fr'{dicomPath}_p.mapping')
cropzone=MP01.cropzone
dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
dirpath=os.path.dirname(dicomPath)
MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,sortBy='seriesNumber')
#MP03 = mapping(data=fr'{dicomPath}_p.mapping')
#Motion correction
MP03.cropzone=cropzone
MP03.go_crop()
MP03.go_resize(scale=2)

dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP02_T2')
MP02 = mapping(data=dicomPath,CIRC_ID=CIRC_ID)
MP02.cropzone=cropzone

MP02.go_crop()
MP02.go_resize(scale=2)
###Test MP02,MP03,TI01
#%%

MP01tmp=MP01._data[:,:,:,0]

datatmp=np.concatenate((MP01tmp[:,:,:,np.newaxis],MP02._data[:,:,:,0:-1],MP03._data),axis=-1)
valueListtmp=['TI 70ms','TE 30ms','TE 35ms','TE 50ms','TE 60ms','TE 70ms','TE 80ms','b50x','b50y','b50z','b500x','b500y','b500z']
map_test=mapping(data=datatmp,valueList=valueListtmp,ID='test_moco',CIRC_ID=CIRC_ID)
Nx,Ny,Nz,Nd=np.shape(datatmp)
# %%
map_test.valueList=valueListtmp
map_test.path=dicomPath
#%%
map_test.imshow_corrected(plot=True)
#%%
moco(map_test._data,map_test)
#%%
from sklearn import preprocessing
from sklearn.cluster import KMeans
X_train=np.reshape(datatmp[:,:,0,:],(Nx*Ny,Nd))
X_train_norm = preprocessing.normalize(X_train,axis=1,norm='l2')
#%%
map_test.imshow_corrected(volume=np.reshape(X_train_norm,(Nx,Ny,1,Nd)),plot=False)
#%%
moco(np.reshape(X_train_norm,(Nx,Ny,1,Nd)),map_test)
# %%
for obj in [map_test]:
    Nz=obj.Nz
    A2=np.copy(obj._data)
    for i in range(Nz):
        A2[:,:,i,:] = obj._data[...,i,:]/np.max(obj._data[...,i,:])*255
    A3=np.vstack((A2[:,:,0,:],A2[:,:,1,:],A2[:,:,2,:]))
    img_dir= os.path.join(os.path.dirname(obj.path),f'{obj.CIRC_ID}_{obj.ID}_moco_.gif')
    obj.createGIF(img_dir,A3,fps=5)
# %%
test_tmp=datatmp[:,:,0,:]
data_regress = np.copy(test_tmp / np.max(test_tmp, axis=(0, 1), keepdims=True))

data_regress=np.reshape(data_regress,(Nx*Ny,Nd))
rpca = R_pca(data_regress)
L, S = rpca.fit(max_iter=10000, iter_print=100)
#%%
L_plot=np.reshape(L,(Nx,Ny,1,Nd))
S_plot=np.reshape(S,(Nx,Ny,1,Nd))
map_test.imshow_corrected(volume=L_plot,ID='L',plot=True)
# %%
map_test.imshow_corrected(volume=S_plot,ID='S',plot=True)

# %%
data_tmp2_0,Transform_2_0 = map_test._coregister_elastix_return_transform(data=np.squeeze(S_plot),target_index=0)
map_test.imshow_corrected(volume=data_tmp2_0[:,:,np.newaxis,:],ID=f'S_plot_naive_Ind0',plot=False)
# %%
import tensorly as tl
data_diff_regress=np.copy(datatmp)
tensor = tl.tensor(data_regress)
tensor_L = tl.tensor(L)
U1 = tl.svd_interface(tensor, n_eigenvecs=10)[0]
U2_diff = tl.svd_interface(tensor_L, n_eigenvecs=1)[2]
U2_diff_resp = tl.svd_interface(tensor, n_eigenvecs=2)[2].T
#%%
G_diff_resp = tl.tenalg.multi_mode_dot( tensor, 
                                    [np.linalg.pinv(U1), np.linalg.pinv(U2_diff_resp)], 
                                    [0,1] )
U2_diff_regress = np.diag( U2_diff.ravel()**(-1) ).dot(  U2_diff_resp )
        # data_diff_regress = U1.dot( G_diff_resp ).dot(U2_diff_regress.T).reshape(data.shape)
data_diff_regress_temp = tl.to_numpy(
                    tl.tucker_to_tensor( (G_diff_resp, (U1, U2_diff_regress)) )
                                        ).reshape((Nx,Ny,Nd))
data_diff_regress=data_diff_regress_temp
#%%
map_test.imshow_corrected(volume=data_diff_regress[:,:,np.newaxis,:],ID=f'data_diff_regress',plot=False)

# %%
#Supposed to be motion only
data_reg=map_test._coregister_elastix_return_transform(data=np.squeeze(S_plot),orig_data=datatmp,target_index=0)
# %%
