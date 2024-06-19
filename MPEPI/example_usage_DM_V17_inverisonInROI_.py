
#%%
from libMapping_v14 import mapping
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
defaultPath= r'C:\Research\MRI\MP_EPI'
#%%
###Hey this won't work with disease!!!
CIRC_ID_List=[446,452,429,419,398,382,381,373,472,498,500]
CIRC_NUMBER=446
CIRC_ID=f'CIRC_00{int(CIRC_NUMBER)}'
print(f'Running{CIRC_ID}')
img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","WITH8000",CIRC_ID)
img_save_dir=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000_ROIFit",CIRC_ID)
if os.path.exists(img_save_dir) is False:
    os.makedirs(img_save_dir)

#Read the MP01-MP03
mapList=[]
#I think I also need the raw data:
#Or I can use the mask from MP02
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)
MP01_0=mapping(mapList[0])
MP01_1=mapping(mapList[1])
MP01_2=mapping(mapList[2])
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            mapList.append(path)
MP01=mapping(mapList[0])
MP02=mapping(mapList[1])
MP03=mapping(mapList[2])

mask=MP02.mask_lv
MP01_list=[MP01_0,MP01_1,MP01_2]

#%%
#Test on the first data set:
import copy
from libMapping_v14 import ir_fit
from numba import *
import time
obj_T1=copy.deepcopy(MP01_0)
global TIlist
TIlist=obj_T1.valueList
data=obj_T1._data

try:
    Nx,Ny,Nd=np.shape(data)
except:
    Nx,Ny,Nz,Nd=np.shape(data)
if len(np.shape(data))==3 or Nz==1:
    NxNy=int(Nx*Ny)
    finalMap=np.zeros(Nx*Ny*1)
    finalRa=np.zeros(Nx*Ny*1)
    finalRb=np.zeros(Nx*Ny*1)
    finalIndex=np.zeros(Nx*Ny*1,dtype=int)
    finalRes=np.zeros(Nx*Ny*1)
elif len(np.shape(data))==4:
    Nx,Ny,Nz,Nd=np.shape(data)
    finalMap=np.zeros(Nx*Ny*Nz)
    finalRa=np.zeros(Nx*Ny*Nz)
    finalRb=np.zeros(Nx*Ny*Nz)
    finalIndex=np.zeros(Nx*Ny*Nz,dtype=int)
    finalRes=np.zeros(Nx*Ny*Nz)
    NxNy=int(Nx*Ny)
tmpmask=mask[:,:,0].reshape(NxNy)
##A faster way is to reshape it into NX*NY*NZ and then use parallel
NxNyNz=Nx*Ny*Nz
dataTmp=np.reshape(data,(NxNyNz,Nd))
##Starting here we would like to use zip repeat argurement.
searchtype='grid'
T1bound=[1,5000]
invertPoint=6
for nn in range(NxNyNz):
    if tmpmask[nn] == True:
        finalMap[nn],finalRa[nn],finalRb[nn],finalIndex[nn],finalRes[nn]=ir_fit(dataTmp[nn,:],TIlist=TIlist,searchtype='grid')

#%%
%matplotlib inline
plt.figure()
ax1=plt.subplot(131)
im1=plt.imshow(finalMap.reshape(Nx,Ny,Nz)[:,:,0],cmap='magma',vmax=3000)
plt.colorbar()
plt.subplot(132)
im1=plt.imshow(finalIndex.reshape(Nx,Ny,Nz)[:,:,0])
plt.colorbar()
plt.subplot(133)
plt.imshow(finalRes.reshape(Nx,Ny,Nz)[:,:,0])
plt.colorbar()

#%%
invertPoint=4
for nn in range(NxNyNz):
    if tmpmask[nn] == True:
        finalMap[nn],finalRa[nn],finalRb[nn],finalIndex[nn],finalRes[nn]=ir_fit(dataTmp[nn,:],TIlist=TIlist,searchtype=searchtype,T1bound=T1bound,invertPoint=invertPoint)

#%%
%matplotlib inline
plt.figure()
ax1=plt.subplot(131)
im1=plt.imshow(finalMap.reshape(Nx,Ny,Nz)[:,:,0],cmap='magma',vmax=3000)
plt.colorbar()
plt.subplot(132)
im1=plt.imshow(finalIndex.reshape(Nx,Ny,Nz)[:,:,0])
plt.colorbar()
plt.subplot(133)
plt.imshow(finalRes.reshape(Nx,Ny,Nz)[:,:,0])
plt.colorbar()

#%%
from libMapping_v14 import sub_ir_fit_grid



dataTmp=np.reshape(data,(NxNyNz,Nd))
for z in range(Nz):
    #finalIndex[mask[:,:]==1]=0
    #The name of count
    u, count = np.unique(finalIndex, return_counts=True)
    print(np.unique(finalIndex, return_counts=True))
    count_sort_ind = np.argsort(-count)
    ##counts = np.bincount(np.reshape(finalIndex[:,:,z],-1))
    #finalIndexSingleSlice=np.argmax(counts)
    #u[count_sort_ind] is the sorted values in the list
    #We need to drop the 0 as this is not in the mask
    if u[count_sort_ind][0]==0:
        finalIndexSingleSlice=u[count_sort_ind][1]
    else:
        finalIndexSingleSlice=u[count_sort_ind][0]
    print(f'Index is',finalIndexSingleSlice)
    try:
        minIndTmp=finalIndexSingleSlice
        invertMatrix=np.concatenate((-np.ones(minIndTmp),np.ones(len(TIlist)-minIndTmp)),axis=0)
        dataTmp=np.reshape(data,(NxNyNz,Nd))
        dataTmp=dataTmp*invertMatrix.T
    except:
        continue

    for nn in range(NxNy):
        finalMap[nn],finalRa[nn],finalRb[nn],finalRes[nn],_=sub_ir_fit_grid(data=dataTmp[nn,:],TIlist=TIlist,T1bound=T1bound)

#%%
%matplotlib inline
plt.figure()
ax1=plt.subplot(131)
im1=plt.imshow(finalMap.reshape(Nx,Ny,Nz)[:,:,0],cmap='magma',vmax=3000)
plt.colorbar()
plt.subplot(132)
im1=plt.imshow(finalRa.reshape(Nx,Ny,Nz)[:,:,0])
plt.colorbar()
plt.subplot(133)
plt.imshow(finalRes.reshape(Nx,Ny,Nz)[:,:,0],vmin=0,vmax=2.6)
plt.colorbar()


#%%
for ss,obj_T1 in enumerate(MP01_list):
    _,_,_,_,_=obj_T1.go_ir_fit_ROI(searchtype='grid',mask=mask[:,:,ss],invertPoint=4,simply=True)
#%%
#Let see the result
#TODOLIST: cancel out the T2 with high value
map_data=np.copy(MP02._map)
map_data[:,:,0]=np.squeeze(MP01_0._map)
map_data[:,:,1]=np.squeeze(MP01_1._map)
map_data[:,:,2]=np.squeeze(MP01_2._map)
MP01._map= np.squeeze(map_data)

MP01.imshow_map(path=img_save_dir,plot=plot)
MP02.imshow_map(path=img_save_dir,plot=plot)
plt.close()
MP03.imshow_map(path=img_save_dir,plot=plot)
#%%
MP01_0.save(filename=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_{MP01_0.ID}_m.mapping'))
MP01_1.save(filename=os.path.join(img_save_dir,f'{MP01_1.CIRC_ID}_{MP01_1.ID}_m.mapping'))
MP01_2.save(filename=os.path.join(img_save_dir,f'{MP01_2.CIRC_ID}_{MP01_2.ID}_m.mapping'))
MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))
MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))


# %%
