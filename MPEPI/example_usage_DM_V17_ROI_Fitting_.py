
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
img_save_dir=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000_NinePoints_Fit",CIRC_ID)
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
def find_closest_indices(list1, list2):
    list2_used = [False] * len(list2)
    closest_indices = []

    for num in list1:
        min_diff = float('inf')
        closest_index = -1
        for i, val in enumerate(list2):
            if not list2_used[i]:
                diff = abs(num - val)
                if diff < min_diff:
                    min_diff = diff
                    closest_index = i
        closest_indices.append(closest_index)
        list2_used[closest_index] = True
    
    return closest_indices

#%%
###Maybe we could keep 110,190,440,940, then 1540, 2140,2700,4000
# 
# ##8 points:
# 
# 
# 
# 10 points: 110, 190, 440,940, 1540, 2140, 2700, 4000, 8000 
list1 = [110, 190, 440, 940, 1540, 2140, 2220, 3000, 4000,8000]
# Get the indices of elements in list1 from list2

for ss,obj_T1 in enumerate(MP01_list):
    valueArray=np.array(obj_T1.valueList)
    closest_indices = find_closest_indices(list1, valueArray)
    print(closest_indices)
    
    closest_indices_not= list(range(len(valueArray)))
    for index in sorted(closest_indices, reverse=True):
        if 0 <= index < len(closest_indices_not):
            closest_indices_not.pop(index)
    print(closest_indices_not)
    #arrayInd=np.where(np.logical_and(valueArray>=700,valueArray<=1200))
    #obj_T1._data[...,arrayInd]=obj_T1._raw_data[...,arrayInd]
    #obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_updated',plot=plot,path=img_save_dir)
    obj_T1._delete(d=closest_indices_not)
    obj_T1.imshow_corrected(ID=f'MP01_Slice{ss}_1_Cropped_updated',plot=plot,path=img_save_dir)





#%%

#Save the stat
stats_file = os.path.join(os.path.dirname(img_save_dir), "MPEPI_stats_v2.csv") 
for obj in MP01_list:
    keys=['CIRC_ID','ID','valueList','shape']
    stats=[obj.CIRC_ID,obj.ID,str(obj.valueList),str(np.shape(obj._data))]
    data=dict(zip(keys,stats))
    cvsdata=pd.DataFrame(data, index=[0])
    if os.path.isfile(stats_file):    
        cvsdata.to_csv(stats_file, index=False, header=False, mode='a')
    else:
        cvsdata.to_csv(stats_file, index=False)



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
