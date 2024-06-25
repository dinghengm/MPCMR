#%%
#This is the file to draw AHA wheel.
#The data in from ims_v2_Feb_5_2024_WITH8000 -> ims_v3_June5_WITH8000


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
plot=False
defaultPath= r'C:\Research\MRI\MP_EPI'

#%%
###Hey this won't work with disease!!!
CIRC_ID_List=[446,452,429,419,398,382,381,373,472,498,500]
#CIRC_NUMBER=446
CIRC_NUMBER=CIRC_ID_List[7]
CIRC_ID=f'CIRC_00{int(CIRC_NUMBER)}'
print(f'Running{CIRC_ID}')

#Here we try to use the clinical map
img_root_dir = os.path.join(defaultPath, "saved_ims_v2_Dec_14_2023")
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            if 'FLASH' in os.path.basename(path) or 'MOLLI' in os.path.basename(path):
                if 'FB' not in os.path.basename(path): 
                    if CIRC_ID in os.path.basename(path): 
                        mapList.append(path)
img_save_dir=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","Clinical Map",CIRC_ID)


if os.path.exists(img_save_dir) is False:
    os.makedirs(img_save_dir)

#print(mapList)

#Clinical T1
map_T1=mapping(mapList[0])
#Clinical T2
map_T2=mapping(mapList[1])

#%%
map_T1.imshow_overlay()
map_T2.imshow_overlay()

#%%
#Get the center
%matplotlib qt
map_T1.go_define_CoMandRVIns()
map_T1.go_AHA_wheel_check()
#%%
#Copy the map from T1 T2 and ADC
map_T1.go_get_AHA_wheel()

map_T2.go_define_CoMandRVIns()
map_T2.go_AHA_wheel_check()

#%%
map_T2.go_get_AHA_wheel()


#%%
for nn,obj in enumerate([map_T1,map_T2]):
    plot=True
    if nn==0:
        txt='T1'
    elif nn==1:
        txt='T2'
    Nx,Ny,Nz=np.shape(obj._map)[0:3]
    figsize = (3.4*Nz, 3)
    fig, axes = plt.subplots(nrows=1, ncols=Nz, figsize=figsize, constrained_layout=True)
    fig.suptitle(f"{txt}", fontsize=16)
    axes=axes.ravel()
    segment_16=obj.segment_16
    maskFinal=np.zeros((Nx,Ny),dtype=int)
    for nn in range(Nz):
        axes[nn].set_axis_off()
        
        if nn<2:
            maskFinal=np.zeros((Nx,Ny),dtype=int)    
            segmentNum =6   
            
            for seg in range(segmentNum):
                maskFinal+=segment_16[nn][seg] * (seg +1)
                
            im = axes[nn].imshow(maskFinal,vmax=6)
        else:
            maskFinal=np.zeros((Nx,Ny),dtype=int)
            segmentNum=4
            #The last slice
            for seg in range(segmentNum):
                maskFinal+=segment_16[nn][seg] * (seg +1)
                
            im = axes[nn].imshow(maskFinal,vmax=4)

    if plot:
        plt.savefig(os.path.join(img_save_dir, f"AHA_16_segments_{txt}.png"))
            
# %%
filename=os.path.join(os.path.dirname(img_save_dir),'mapping_AHA.csv')
map_T1.export_stats(filename=filename,crange=[0,5000])

map_T2.export_stats(filename=filename,crange=[0,150])

# %%
map_T1.save(filename=os.path.join(img_save_dir,f'{map_T1.CIRC_ID}_{map_T1.ID}_p.mapping'))
map_T2.save(filename=os.path.join(img_save_dir,f'{map_T2.CIRC_ID}_{map_T2.ID}_p.mapping'))

# %%
