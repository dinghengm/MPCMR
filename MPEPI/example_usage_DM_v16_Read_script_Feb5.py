#########################
#########This is the script to generate the maps in batch (5 mins per subject)
######### This time, I used the m.mapping that already have before
######### I am going to test four things: 1. with 8000, how is the maps
###############2. if without 8000 how is the maps
##############3. if have >5000, remove 8000 how is the maps.
######### 2. without 700-1100 how is the maps
########  3. Generate the maps with 700-1100 as original, and else corrected
########  4. Generate the maps with the raw data
#########FROM saved_ims
#########TO   saved_ims_v2_Feb_5_2024
#########SUBJECTS ['446','452','429','419','407','405','398','382','381','373','457','471','472','486','498','500']
#########MAPSAVEDAS m.mapping
#%%

from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
matplotlib.rcParams['savefig.dpi'] = 400
plot=True
import h5py

#%%
#for id_idx in range(len(CIRC_ID_list)): #[0]:
def update_data(CIRC_NUMBER):
#for id_idx in [-1]:
    #CIRC_NUMBER=CIRC_ID_list[id_idx]
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
    #Get the shape of all data, and then replace the data with corrected
    #Read the data
    #Renew the dataset:
    img_moco_dir='C:\Research\MRI\MP_EPI\Moco_Dec6\MOCO'
    moco_data=h5py.File(os.path.join(img_moco_dir,rf'{CIRC_ID}_MOCO.mat'),'r')

    for ss,obj_T1 in enumerate(MP01_list):
        key=f'moco_Slice{ss}'
        moco_data_single_slice=np.transpose(moco_data[key],(2,1,0))
        Ndtmp=0
        for obj in MPs_list:
            if 'mp01' in obj.ID.lower():
                Ndtmp_end=np.shape(MP01_list[ss]._data)[-1]
                obj_T1._update_data(moco_data_single_slice[:,:,0:Ndtmp_end])
                print(obj_T1.ID,np.shape(obj_T1._data))
                print('valueList=',obj_T1.valueList)
            else:
                Ndtmp_start=Ndtmp_end
                Ndtmp_end+=np.shape(obj._data)[-1]
                obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
                print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
                #print('valueList=',obj.valueList)
    
    
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
    

    #with 8000, how is the maps
    saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","WITH8000",CIRC_ID)
    stats_file = os.path.join(os.path.dirname(saved_img_root_dir), "MPEPI_stats_v2.csv") 
    if not os.path.exists(saved_img_root_dir):
        os.mkdir(saved_img_root_dir)
    for ss,obj_T1 in enumerate(MP01_list):
        valueArray=np.array(obj_T1.valueList)
        arrayInd=np.where(np.logical_and(valueArray>=700,valueArray<=1200))
        obj_T1._data[...,arrayInd]=obj_T1._raw_data[...,arrayInd]
        if np.size(np.where(np.logical_and(valueArray>=4500,valueArray<=8000)))==0:
            if np.size(np.where(valueArray>=8000))!=1:
                pass
            else:
                data1=MP02._raw_data[:,:,ss,0]
                data1=np.concatenate((np.squeeze(obj_T1._data),data1[:,:,np.newaxis]),axis=-1)
                obj_T1.valueList.append(8000)
                obj_T1._update_data(data1[:,:,np.newaxis,:])
        obj_T1._update()
        obj_T1.go_crop_Auto()
        obj_T1.go_resize()
        obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_with8000',plot=plot,path=saved_img_root_dir)
        plt.close()
        obj_T1.save(filename=os.path.join(saved_img_root_dir,f'{obj_T1.CIRC_ID}_{obj_T1.ID}_m.mapping'))
        keys=['CIRC_ID','ID','valueList','shape']
        stats=[obj_T1.CIRC_ID,obj_T1.ID,str(obj_T1.valueList),str(np.shape(obj_T1._data)),obj_T1.cropzone]
        data=dict(zip(keys,stats))
        cvsdata=pd.DataFrame(data, index=[0])
        if os.path.isfile(stats_file):    
            cvsdata.to_csv(stats_file, index=False, header=False, mode='a')
        else:
            cvsdata.to_csv(stats_file, index=False)
    MP02.go_crop_Auto()
    MP02.go_resize()
    MP02.imshow_corrected(ID=f'MP02_with8000',plot=plot,path=saved_img_root_dir)
    plt.close()
    MP03.go_crop_Auto()
    MP03.go_resize()
    MP03.imshow_corrected(ID=f'MP03_with8000',plot=plot,path=saved_img_root_dir,valueList=MP03.bval)
    plt.close()    
    MP01.save(filename=os.path.join(saved_img_root_dir,f'{MP01.CIRC_ID}_{MP01.ID}_m.mapping'))
    MP02.save(filename=os.path.join(saved_img_root_dir,f'{MP02.CIRC_ID}_{MP02.ID}_m.mapping'))
    MP03.save(filename=os.path.join(saved_img_root_dir,f'{MP03.CIRC_ID}_{MP03.ID}_m.mapping'))

#Raw images without the moco
    saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","RAW",CIRC_ID)
    if not os.path.exists(saved_img_root_dir):
        os.mkdir(saved_img_root_dir)
    for ss,obj_T1 in enumerate(MP01_list):
        obj_T1._data=np.copy(obj_T1._raw_data)
        obj_T1._update()
        if len(obj_T1.valueList) != np.shape(obj_T1._data)[-1]:
            obj_T1.valueList.pop()
        obj_T1.go_crop_Auto()
        obj_T1.go_resize()
        obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_raw',plot=plot,path=saved_img_root_dir)
        plt.close()
        obj_T1.save(filename=os.path.join(saved_img_root_dir,f'{obj.CIRC_ID}_{obj.ID}_m.mapping'))

print("Done Cropping!")
    
# %%

if __name__=='__main__':
    from multiprocessing import Pool
    CIRC_ID_List=[446,452,429,419,407,405,398,382,381,373,457,471,472,486,498,500]
    with Pool(5) as p:  # Create a pool of 5 processes
        results = p.map(update_data,CIRC_ID_List)
