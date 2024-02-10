#########################
#########This is the script to generate the maps in batch (5 mins per subject)
#########FROM saved_ims
#########TO   saved_ims_v2_Jan_12_2024
#########SUBJECTS ['452','457','471','472','486','498','500']
#########MAPSAVEDAS m.mapping







from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v13 import readFolder,decompose_LRT,moco,moco_naive
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk # conda pip install SimpleITK-SimpleElastix
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
from imgbasics import imcrop
import pandas as pd
from skimage.transform import resize as imresize
import h5py
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib

matplotlib.rcParams['savefig.dpi'] = 400
plot=True



def update_map(CIRC_NUMBER):
    CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
    img_root_dir = os.path.join(defaultPath, "saved_ims",CIRC_ID)
    saved_img_root_dir=os.path.join(defaultPath, "saved_ims_v2_Jan_12_2024",CIRC_ID)
    if not os.path.exists(saved_img_root_dir):
                os.mkdir(saved_img_root_dir)

    # image root directory
    # Statas saved 
    stats_file = os.path.join(defaultPath, "MPEPI_stats_v16.csv") 

    #Read the MP01-MP03
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('.mapping'):
                if path.endswith('p.mapping')==False and path.endswith('m.mapping')==False:
                    mapList.append(path)
    print(mapList)

    img_moco_dir='C:\Research\MRI\MP_EPI\Moco_Dec6\MOCO'
    moco_data=h5py.File(os.path.join(img_moco_dir,rf'{CIRC_ID}_MOCO.mat'),'r')

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
    for ss,obj_T1 in enumerate(MP01_list):
        key=f'moco_Slice{ss}'
        moco_data_single_slice=np.transpose(moco_data[key],(2,1,0))
        Ndtmp=0
        for obj in MPs_list:
            if 'mp01' in obj.ID.lower():
                Ndtmp_end=np.shape(MP01_list[ss]._data)[-1]
                obj_T1._update_data(moco_data_single_slice[:,:,np.newaxis,0:Ndtmp_end])
                obj_T1.go_crop_Auto()
                obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped',plot=plot,path=saved_img_root_dir)
                obj_T1.go_create_GIF(path_dir=str(saved_img_root_dir))
                obj_T1._update_data(moco_data_single_slice[:,:,0:Ndtmp_end])
                print(obj_T1.ID,np.shape(obj_T1._data))
                print('valueList=',obj_T1.valueList)
            else:
                Ndtmp_start=Ndtmp_end
                Ndtmp_end+=np.shape(obj._data)[-1]
                obj._data[:,:,ss,:]=moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]
                print(obj.ID,np.shape(moco_data_single_slice[:,:,Ndtmp_start:Ndtmp_end]))
                #print('valueList=',obj.valueList)
    '''
    for obj in MP01_list:
        obj.go_crop_Auto()
        obj.go_resize()
    for obj in MPs_list:
        obj.go_crop_Auto()
        obj.go_resize()
    '''
    #%%
    #Replace the between frames with the original frames
    #Conservative in 800-900 only
    for ss,obj_T1 in enumerate(MP01_list):
        valueArray=np.array(obj_T1.valueList)
        arrayInd=np.where(np.logical_and(valueArray>=700,valueArray<=1200))
        obj_T1._data[...,arrayInd]=obj_T1._raw_data[...,arrayInd]
        obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_updated',plot=plot,path=saved_img_root_dir)

    #Save the file as M
    stats_file = os.path.join(os.path.dirname(saved_img_root_dir), "MPEPI_stats_v2.csv") 

    for obj in MP01_list:
        obj.save(filename=os.path.join(saved_img_root_dir,f'{obj.CIRC_ID}_{obj.ID}_m.mapping'))
        keys=['CIRC_ID','ID','valueList','shape']
        stats=[obj.CIRC_ID,obj.ID,str(obj.valueList),str(np.shape(obj._data))]
        data=dict(zip(keys,stats))
        cvsdata=pd.DataFrame(data, index=[0])
        if os.path.isfile(stats_file):    
            cvsdata.to_csv(stats_file, index=False, header=False, mode='a')
        else:
            cvsdata.to_csv(stats_file, index=False)

    for obj in MPs_list:
        obj.save(filename=os.path.join(saved_img_root_dir,f'{obj.CIRC_ID}_{obj.ID}_m.mapping'))
        keys=['CIRC_ID','ID','valueList']
        if 'mp02' in obj.ID.lower():
            stats=[obj.CIRC_ID,obj.ID,str(obj.valueList)]
        elif 'mp03' in obj.ID.lower():
            stats=[obj.CIRC_ID,obj.ID,str(obj.bval)]
        else:
            continue
        data=dict(zip(keys,stats))
        cvsdata=pd.DataFrame(data, index=[0])
        if os.path.isfile(stats_file):    
            cvsdata.to_csv(stats_file, index=False, header=False, mode='a')
        else:
            cvsdata.to_csv(stats_file, index=False)

    img_save_dir=os.path.join(defaultPath, "saved_ims_v2_Jan_12_2024",CIRC_ID)
    print(f'Running{CIRC_ID}')
    img_root_dir = img_save_dir
    #img_save_dir=os.path.join(defaultPath,'saved_ims_v2_Jan_12_2024')
    #Read the MP01-MP03
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('m.mapping'):
                mapList.append(path)
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
        #obj._data=np.copy(obj._raw_data)
        #obj.go_resize()
        obj._update()
        obj.imshow_corrected(ID=f'{obj.ID}_moco',plot=plot)
    for obj in MP01_list:
        #obj._data=np.copy(obj._raw_data)
        #obj.go_resize()
        obj._update()
    for ss,obj_T1 in enumerate(MP01_list):
        finalMap,finalRa,finalRb,finalRes=obj_T1.go_ir_fit(searchtype='grid',invertPoint=4)
        plt.figure()
        plt.axis('off')
        plt.imshow(finalMap.squeeze(),cmap='magma',vmin=0,vmax=3000)
        img_dir= os.path.join(img_save_dir,f'{obj.CIRC_ID}_MP01_Slice{ss}_T1')
        plt.savefig(img_dir)
        plt.close()
        obj_T1._map=finalMap
        #obj_T1.save(filename=os.path.join(img_root_dir,f'{obj_T1.ID}_p.mapping'))

    MP02._update()
    #MP02.imshow_corrected(ID=f'MP02_Corrected',plot=plot,path=img_root_dir)
    plt.close()
    MP03._update()
    #MP03.imshow_corrected(ID=f'MP03_Corrected',valueList=MP03.bval,plot=plot,path=img_root_dir)
    plt.close()
    MP03.go_cal_ADC()
    MP02.go_t2_fit()
    map_data=np.copy(MP02._map)
    map_data[:,:,0]=np.squeeze(MP01_0._map)
    map_data[:,:,1]=np.squeeze(MP01_1._map)
    map_data[:,:,2]=np.squeeze(MP01_2._map)
    MP01._map= np.squeeze(map_data)

    MP01.imshow_map(path=img_save_dir,plot=plot)
    plt.close()
    MP02.imshow_map(path=img_save_dir,plot=plot)
    plt.close()
    MP03.imshow_map(path=img_save_dir,plot=plot)
    plt.close()
    MP01_0.save(filename=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_{MP01_0.ID}_m.mapping'))
    MP01_1.save(filename=os.path.join(img_save_dir,f'{MP01_1.CIRC_ID}_{MP01_1.ID}_m.mapping'))
    MP01_2.save(filename=os.path.join(img_save_dir,f'{MP01_2.CIRC_ID}_{MP01_2.ID}_m.mapping'))
    MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_m.mapping'))
    MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_m.mapping'))
    MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_m.mapping'))



if __name__=='__main__':

    from multiprocessing import Pool
    
    #CIRC_ID_List=['446','452','429','419','407','405','398','382','381','373']
    CIRC_ID_List=['452','457','471','472','486','498','500']
    #CIRC_ID_List=['429','398']
    with Pool(5) as p:  # Create a pool of 5 processes
        results = p.map(update_map,CIRC_ID_List)
    