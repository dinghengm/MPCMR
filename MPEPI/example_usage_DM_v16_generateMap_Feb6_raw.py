#########################
#########This is the script to generate the maps in batch (5 mins per subject)
#########FROM saved_ims
#########TO   saved_ims_v2_Dec_14_2023
#########SUBJECTS '446','452','429','419','407','405','398','382','381','373'
#########MAPSAVEDAS m.mapping



import argparse
import sys
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
import pandas as pd
import h5py
import warnings #we know deprecation may show bc we are using a stable older ITK version
defaultPath= r'C:\Research\MRI\MP_EPI'
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
def go_generate_maps(CIRC_NUMBER):
    CIRC_ID=f'CIRC_00{int(CIRC_NUMBER)}'
    img_root_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)
    # image root directory

    #Read the MP01-MP03
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('.mapping'):
                if path.endswith('p.mapping')==False and path.endswith('m.mapping')==False:
                    mapList.append(path)
    print(mapList)
    MP01_0=mapping(mapList[0])
    MP01_1=mapping(mapList[1])
    MP01_2=mapping(mapList[2])
    MP01=mapping(mapList[3])
    MP02=mapping(mapList[4])
    #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
    #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
    MP03=mapping(mapList[5])

    MP01_list=[MP01_0,MP01_1,MP01_2]
    MPs_list=[MP01,MP02,MP03]

    for obj in MPs_list:
        obj._data=np.copy(obj._raw_data)
    for obj in MP01_list:
        obj._data=np.copy(obj._raw_data)

    if CIRC_NUMBER==452:
        MP01_0._delete(d=[4,5,6,-2])
        MP01_1._delete(d=[5,6,7])
        MP01_2._delete(d=[5])
    elif CIRC_NUMBER==446:
        MP01_0._delete(d=[5,-1])
        MP01_1._delete(d=[-1])
        MP01_2._delete(d=[4,-1])
    elif CIRC_NUMBER==407:
        MP01_0._delete(d=[-1])
        MP01_1._delete(d=[-1])
        MP01_2._delete(d=[-1])
    elif CIRC_NUMBER==398:
        MP01_0._delete(d=[4])
        MP01_1._delete(d=[5])
        #MP01_2._delete(d=[-1])
    elif CIRC_NUMBER==472:
        MP01_0._delete(d=[0,3,7,-1])
        MP01_1._delete(d=[5,7,8,13,-1])
        MP01_2._delete(d=[1,8])
    elif CIRC_NUMBER==486:
        #MP01_0._delete(d=[4])
        MP01_1._delete(d=[1,4])
        #MP01_2._delete(d=[-1])
    elif CIRC_NUMBER==498 :
        MP01_0._delete(d=[-6,-1])
    elif CIRC_NUMBER==500:
        MP01_0._delete(d=[4,-1])
        MP01_1._delete(d=[2])
        #MP01_2._delete(d=[-1])
        
    saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","RAW",CIRC_ID)
    if not os.path.exists(saved_img_root_dir):
        os.mkdir(saved_img_root_dir)
    for ss,obj_T1 in enumerate(MP01_list):
        valueArray=np.array(obj_T1.valueList)
        obj_T1._data=np.copy(obj_T1._raw_data)
        obj_T1._update()
        if np.size(np.where(np.logical_and(valueArray>=4500,valueArray<=8000)))==0:
            if np.size(np.where(valueArray>=8000))!=1:
                pass
            else:
                data1=MP02._raw_data[:,:,ss,0]
                data1=np.concatenate((np.squeeze(obj_T1._data),data1[:,:,np.newaxis]),axis=-1)
                obj_T1.valueList.append(8000)
                obj_T1._update_data(data1[:,:,np.newaxis,:])
        obj_T1.go_crop_Auto()
        obj_T1.go_resize()
        obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_raw',plot=plot,path=saved_img_root_dir)
        plt.close()
        obj_T1.save(filename=os.path.join(saved_img_root_dir,f'{obj.CIRC_ID}_{obj.ID}_m.mapping'))

    for ss,obj_T1 in enumerate(MP01_list):
        finalMap,finalRa,finalRb,finalRes=obj_T1.go_ir_fit(searchtype='grid',invertPoint=4)
        plt.figure()
        plt.axis('off')
        plt.imshow(finalMap.squeeze(),cmap='magma',vmin=0,vmax=3000)
        img_dir= os.path.join(saved_img_root_dir,f'{obj_T1.CIRC_ID}_MP01_Slice{ss}_T1')
        plt.savefig(img_dir)
        plt.close()
        obj_T1._map=finalMap
        obj_T1.save(filename=os.path.join(saved_img_root_dir,f'{obj_T1.ID}_m.mapping'))
    Nx,Ny,_,_=np.shape(MP01_0._data)
    map_data=np.zeros((Nx,Ny,3))
    map_data[:,:,0]=np.squeeze(MP01_0._map)
    map_data[:,:,1]=np.squeeze(MP01_1._map)
    map_data[:,:,2]=np.squeeze(MP01_2._map)
    MP01._map= np.squeeze(map_data)

    MP01.imshow_map(path=saved_img_root_dir,plot=plot)
    plt.close()

    MP01_0.save(filename=os.path.join(saved_img_root_dir,f'{MP01_0.CIRC_ID}_{MP01_0.ID}_m.mapping'))
    MP01_1.save(filename=os.path.join(saved_img_root_dir,f'{MP01_1.CIRC_ID}_{MP01_1.ID}_m.mapping'))
    MP01_2.save(filename=os.path.join(saved_img_root_dir,f'{MP01_2.CIRC_ID}_{MP01_2.ID}_m.mapping'))
    MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_m.mapping'))


if __name__=='__main__':
    from multiprocessing import Pool
    CIRC_ID_List=[446,452,429,419,407,405,398,382,381,373,457,471,472,486,498,500]
    with Pool(5) as p:  # Create a pool of 5 processes
        results = p.map(go_generate_maps,CIRC_ID_List)

    