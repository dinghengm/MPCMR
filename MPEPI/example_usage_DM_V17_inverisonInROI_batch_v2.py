
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
def go_generate_maps(CIRC_NUMBER):
    CIRC_ID=f'CIRC_00{int(CIRC_NUMBER)}'
    print(f'Running{CIRC_ID}')
    img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Feb_5_2024","WITH8000",CIRC_ID)
    #img_save_dir=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000",CIRC_ID)

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
    for ss,obj_T1 in enumerate(MP01_list):
        _,_,_,_,_=obj_T1.go_ir_fit_ROI(searchtype='grid',mask=mask[:,:,ss],invertPoint=4,simply=True)

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
    plt.close()
    MP01_0.save(filename=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_{MP01_0.ID}_m.mapping'))
    MP01_1.save(filename=os.path.join(img_save_dir,f'{MP01_1.CIRC_ID}_{MP01_1.ID}_m.mapping'))
    MP01_2.save(filename=os.path.join(img_save_dir,f'{MP01_2.CIRC_ID}_{MP01_2.ID}_m.mapping'))
    MP02.save(filename=os.path.join(img_save_dir,f'{MP02.CIRC_ID}_{MP02.ID}_p.mapping'))
    MP03.save(filename=os.path.join(img_save_dir,f'{MP03.CIRC_ID}_{MP03.ID}_p.mapping'))
    MP01.save(filename=os.path.join(img_save_dir,f'{MP01.CIRC_ID}_{MP01.ID}_p.mapping'))



if __name__=='__main__':
    from multiprocessing import Pool

    #CIRC_ID_List=[446,452,429,419,407,405,398,382,381,373,457,471,472,486,498,500]
    CIRC_ID_List=[446,452,429,419,398,382,381,373,472,498,500]

    with Pool(10) as p:  # Create a pool of 5 processes
        results = p.map(go_generate_maps,CIRC_ID_List)
    #go_generate_maps(446)


# %%
