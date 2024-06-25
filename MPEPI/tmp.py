#%%
####################Currently I am testing if I could get click on the center of mass and also the edge of 
####################
from libMapping_v14 import *  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os,sys
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
from CIRC_tools import *
matplotlib.rcParams['savefig.dpi'] = 400
plot=False

#%%
#CIRC_ID_List=['446','452','429','419','407','405','398','382','381','373']
#CIRC_NUMBER=CIRC_ID_List[9]
CIRC_ID_List=['500','498','472','446','452','429','419','398','382','381','373']
CIRC_NUMBER=CIRC_ID_List[2]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
img_root_dir = os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000",CIRC_ID)

#img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Jan_12_2024')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('p.mapping'):
            if CIRC_ID in os.path.basename(path):
                mapList.append(path)
map_T1=mapping(mapList[0])
map_T2=mapping(mapList[1])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
map_DWI=mapping(mapList[2])
#%%
def bullsEye(valuesPlot,valuesOri):
    from PIL import Image, ImageDraw
    #base_image = Image.new("RGB", (500, 500), (255, 255, 255))
    #canvas = Image.new("RGB", (500, 500), (255, 255, 255))
    canvas = Image.new('L', (500,500), (0))
    draw = ImageDraw.Draw(canvas)
    draw.arc([100, 100, 400, 400], 0, 60, fill=int(valuesPlot[4]), width=40)
    draw.arc([100, 100, 400, 400], 60, 120, fill=int(valuesPlot[3]), width=40)
    draw.arc([100, 100, 400, 400], 120, 180, fill=int(valuesPlot[2]), width=40)
    draw.arc([100, 100, 400, 400], 180, 240, fill=int(valuesPlot[1]), width=40)
    draw.arc([100, 100, 400, 400], 240, 300, fill=int(valuesPlot[0]), width=40)
    draw.arc([100, 100, 400, 400], 300, 0, fill=int(valuesPlot[5]), width=40)

    draw.arc([145, 145, 355, 355], 0, 60, fill=int(valuesPlot[10]), width=40)
    draw.arc([145, 145, 355, 355], 60, 120, fill=int(valuesPlot[9]), width=40)
    draw.arc([145, 145, 355, 355], 120, 180, fill=int(valuesPlot[8]), width=40)
    draw.arc([145, 145, 355, 355], 180, 240, fill=int(valuesPlot[7]), width=40)
    draw.arc([145, 145, 355, 355], 240, 300, fill=int(valuesPlot[6]), width=40)
    draw.arc([145, 145, 355, 355], 300, 0, fill=int(valuesPlot[11]), width=40)

    draw.arc([190, 190, 310, 310], 45, 135, fill=int(valuesPlot[14]), width=40)
    draw.arc([190, 190, 310, 310], 135, 225, fill=int(valuesPlot[13]), width=40)
    draw.arc([190, 190, 310, 310], 225, 315, fill=int(valuesPlot[12]), width=40)
    draw.arc([190, 190, 310, 310], 315, 45, fill=int(valuesPlot[15]), width=40)

    #draw.arc([235, 235, 265, 265], 1, 0, fill=int(values[16]), width=40)

    points = [(225, 120), (125, 180), (125, 315), (230, 375), (325, 340), (345, 170), (225, 165), (155, 210), (155, 280), (220, 330), (310, 280), (315, 215), (240, 210), (195, 250), (235, 285), (275, 250), (236, 247)]
    
    #alpha=0.5
    #blended_image = Image.blend(base_image, canvas, alpha)

    for i in range(16):
        level = valuesOri[i]
        print(str(i+1)+". Pixel value: "+str(int(valuesOri[i]))+" and Level: "+str(level))
        draw.text(points[i], str(round(level,1)))

    del draw
    output = np.array(canvas)
    return output
#%%
saveAHA_Path=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000","AHA_SingleMap")

def generateSingleAHA(obj,path,crange,cmap):
    new_min=0
    new_max=255
    segmentValueAveList=[]
    segmentValueStdList=[]
    segment_16=obj.segment_16
    for nn,segment_stack in enumerate(segment_16):
                            #Loop over values in each slice [6, 6, 4]

        for segment in segment_stack:
            #mask=map_T1.mask_lv[:,:,nn].squeeze()
            map=obj._map[:,:,nn].squeeze()
            segmentValueAveList.append(np.mean(map[segment]))
            segmentValueStdList.append(np.std(map[segment]))
    crange=crange

    segmentValueAveListPlot=[ new_min + (value - crange[0]) * (new_max - new_min) / (crange[1] - crange[0])
        for value in segmentValueAveList]
    output=bullsEye(segmentValueAveListPlot,segmentValueAveList)
    '''outputPlot=[ crange[0] + (value - 0) * (crange[1] - crange[0]) / (255 - 0)
        for value in output]'''
    #plt.imshow(outputPlot,vmax=crange[1],cmap='magma')
    plt.imshow(output,vmax=255,cmap=cmap)
    plt.axis('off')
    #plt.colorbar()
    aha_img_save_path=os.path.join(saveAHA_Path,f'{obj.CIRC_ID}_{obj.ID}_AHA')
    if os.path.exists(saveAHA_Path) is False: 
        os.makedirs(saveAHA_Path)
    plt.savefig(aha_img_save_path,bbox_inches='tight')

#%%

crangeList=[[800,1500],[20,60],[0.5,2.5]]
cmapList=['magma','viridis','hot']objList=[map_T1,map_T2,map_DWI]
for num,obj in enumerate(objList):
    generateSingleAHA(obj,saveAHA_Path,crangeList[num],cmapList[num])

#%%
#For clinical map
crangeList=[[800,1500],[20,60]]
cmapList=['magma','viridis']
objList=[map_T1,map_T2]
for num,obj in enumerate(objList):
    generateSingleAHA(obj,saveAHA_Path,crangeList[num],cmapList[num])
#%%
saveAHA_Path=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","Clinical Map","AHA_SingleMap")

CIRC_ID_List=['500','498','472','446','452','429','419','398','382','381','373']
for CIRC_NUMBER in CIRC_ID_List:
    #CIRC_NUMBER=CIRC_ID_List[2]
    CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
    print(f'Running{CIRC_ID}')
    img_root_dir = os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000",CIRC_ID)

    #img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Jan_12_2024')
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('p.mapping'):
                if CIRC_ID in os.path.basename(path):
                    mapList.append(path)
    map_T1=mapping(mapList[0])
    map_T2=mapping(mapList[1])
    #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
    #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
    map_DWI=mapping(mapList[2])
    crangeList=[[800,1500],[20,60],[0.5,2.5]]
    cmapList=['magma','viridis','hot']
    objList=[map_T1,map_T2,map_DWI]

    try:
        for num,obj in enumerate(objList):
            generateSingleAHA(obj,saveAHA_Path,crangeList[num],cmapList[num])
    except:
        print(f'Error in {CIRC_ID}')

#%%
#Global read value in the excel sheet:
import pandas as pd
import os
import scipy
dirname=r'C:\Research\MRI\MP_EPI\saved_ims_v3_June_5_2024\WITH8000'
df=pd.read_csv(os.path.join(dirname,r'mapping_AHA.csv'))
CIRC_ID_list=['CIRC_00373','CIRC_00381','CIRC_00382','CIRC_00398','CIRC_00405','CIRC_00419', 'CIRC_00429','CIRC_00446','CIRC_00472','CIRC_00486','CIRC_00498','CIRC_00500']    
ID_list=['MP01','MP02','MP03','T1_MOLLI','T1_MOLLI_FB','T2_FLASH','T2_FLASH_FB']
df_CIRD=df[df['CIRC_ID'].str.contains('|'.join(CIRC_ID_list))]
df_CIRD=df
#%%
index=2
df_copy=df_CIRD.copy()
searchfor_T1=[ID_list[i] for i in [index]]
df_copy=df_copy[df_copy['ID'].str.contains('|'.join(searchfor_T1),case=False)]
keys_aha=['Basal Anterior', 'Basal Anteroseptal', 'Basal Inferoseptal',
               'Basal Inferior', 'Basal Inferolateral', 'Basal Anterolateral',
               'Mid Anterior', 'Mid Anteroseptal', 'Mid Inferoseptal',
               'Mid Inferior', 'Mid Inferolateral', 'Mid Anterolateral',
               'Apical Anterior', 'Apical Septal', 
               'Apical Inferior', 'Apical Lateral']
segmentValueAveList=[]
for y_ind,str_read in enumerate(keys_aha):
    df_tmp_key=df_copy[str_read]
    df_mean=df_tmp_key.mean()
    segmentValueAveList.append(df_mean)


crange=crangeList[index]

segmentValueAveListPlot=[ new_min + (value - crange[0]) * (new_max - new_min) / (crange[1] - crange[0])
    for value in segmentValueAveList]
output=bullsEye(segmentValueAveListPlot,segmentValueAveList)
'''outputPlot=[ crange[0] + (value - 0) * (crange[1] - crange[0]) / (255 - 0)
    for value in output]'''
%matplotlib inline
#plt.imshow(outputPlot,vmax=crange[1],cmap='magma')
plt.imshow(output,vmax=255,cmap=cmapList[index])
plt.axis('off')
#plt.colorbar()
aha_img_save_path=os.path.join(saveAHA_Path,f'{ID_list[index]}_{len(df_copy)}_AHA')
if os.path.exists(saveAHA_Path) is False: 
    os.makedirs(saveAHA_Path)
plt.savefig(aha_img_save_path)
#%%
#################Clinical Map
CIRC_ID_List=['500','498','472','446','452','429','419','398','382','381','373']
for CIRC_NUMBER in CIRC_ID_List:
    #CIRC_NUMBER=CIRC_ID_List[2]
    CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
    print(f'Running{CIRC_ID}')
    img_root_dir = os.path.join(defaultPath, "saved_ims_v3_June_5_2024","Clinical Map",CIRC_ID)

    #img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Jan_12_2024')
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('p.mapping'):
                if CIRC_ID in os.path.basename(path):
                    mapList.append(path)
    try:
        map_T1=mapping(mapList[0])
        map_T2=mapping(mapList[1])
        #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
        #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
        #map_DWI=mapping(mapList[2])
        crangeList=[[800,1500],[20,60]]
        cmapList=['magma','viridis']
        objList=[map_T1,map_T2]
        for num,obj in enumerate(objList):
            generateSingleAHA(obj,saveAHA_Path,crangeList[num],cmapList[num])
    except:
        print(f'Error in {CIRC_ID}')

#%%
##########Clinical Map
dirname=r'C:\Research\MRI\MP_EPI\saved_ims_v3_June_5_2024\Clinical Map'
df=pd.read_csv(os.path.join(dirname,r'mapping_AHA.csv'))
CIRC_ID_list=['CIRC_00373','CIRC_00381','CIRC_00382','CIRC_00398','CIRC_00405','CIRC_00419', 'CIRC_00429','CIRC_00446','CIRC_00472','CIRC_00486','CIRC_00498','CIRC_00500']    
ID_list=['T1_MOLLI','T2_FLASH']
df_CIRD=df[df['CIRC_ID'].str.contains('|'.join(CIRC_ID_list))]
df_CIRD=df
#%%
index=1
df_copy=df_CIRD.copy()
searchfor_T1=[ID_list[i] for i in [index]]
df_copy=df_copy[df_copy['ID'].str.contains('|'.join(searchfor_T1),case=False)]
keys_aha=['Basal Anterior', 'Basal Anteroseptal', 'Basal Inferoseptal',
               'Basal Inferior', 'Basal Inferolateral', 'Basal Anterolateral',
               'Mid Anterior', 'Mid Anteroseptal', 'Mid Inferoseptal',
               'Mid Inferior', 'Mid Inferolateral', 'Mid Anterolateral',
               'Apical Anterior', 'Apical Septal', 
               'Apical Inferior', 'Apical Lateral']
segmentValueAveList=[]
for y_ind,str_read in enumerate(keys_aha):
    df_tmp_key=df_copy[str_read]
    df_mean=df_tmp_key.mean()
    segmentValueAveList.append(df_mean)


crange=crangeList[index]

segmentValueAveListPlot=[ new_min + (value - crange[0]) * (new_max - new_min) / (crange[1] - crange[0])
    for value in segmentValueAveList]
output=bullsEye(segmentValueAveListPlot,segmentValueAveList)
'''outputPlot=[ crange[0] + (value - 0) * (crange[1] - crange[0]) / (255 - 0)
    for value in output]'''
%matplotlib inline
#plt.imshow(outputPlot,vmax=crange[1],cmap='magma')
plt.imshow(output,vmax=255,cmap=cmapList[index])
plt.axis('off')
#plt.colorbar()
aha_img_save_path=os.path.join(saveAHA_Path,f'{ID_list[index]}_{len(df_copy)}_AHA')
if os.path.exists(saveAHA_Path) is False: 
    os.makedirs(saveAHA_Path)
plt.savefig(aha_img_save_path)



#%%
#AHA implementation:
#Normalized to 0-1000
saveAHA_T1_Path=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000","AHA_SingleMap")
new_min=0
new_max=255
segmentValueAveList=[]
segmentValueStdList=[]
segment_16=map_T1.segment_16
for nn,segment_stack in enumerate(segment_16):
                        #Loop over values in each slice [6, 6, 4]

    for segment in segment_stack:
        #mask=map_T1.mask_lv[:,:,nn].squeeze()
        map=map_T1._map[:,:,nn].squeeze()
        segmentValueAveList.append(np.mean(map[segment]))
        segmentValueStdList.append(np.std(map[segment]))


crange=[800,1500]

segmentValueAveListPlot=[ new_min + (value - crange[0]) * (new_max - new_min) / (crange[1] - crange[0])
    for value in segmentValueAveList]
output=bullsEye(segmentValueAveListPlot,segmentValueAveList)
'''outputPlot=[ crange[0] + (value - 0) * (crange[1] - crange[0]) / (255 - 0)
    for value in output]'''
%matplotlib qt
#plt.imshow(outputPlot,vmax=crange[1],cmap='magma')
plt.imshow(output,vmax=255,cmap='magma')
plt.axis('off')
#plt.colorbar()
aha_img_save_path=os.path.join(saveAHA_T1_Path,f'{map_T1.CIRC_ID}_{map_T1.ID}_AHA')
if os.path.exists(saveAHA_T1_Path) is False: 
    os.makedirs(saveAHA_T1_Path)
plt.savefig(aha_img_save_path)
#cv2.imwrite(dirName+'/Bulls_Eye.png', output)
#cv2.imwrite(dirName+'/Stacked_Output.png', stack)


#%%
saveAHA_Path=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000","AHA_SingleMap")
new_min=0
new_max=255
segmentValueAveList=[]
segmentValueStdList=[]
segment_16=map_T2.segment_16
for nn,segment_stack in enumerate(segment_16):
                        #Loop over values in each slice [6, 6, 4]

    for segment in segment_stack:
        #mask=map_T1.mask_lv[:,:,nn].squeeze()
        map=map_T2._map[:,:,nn].squeeze()
        segmentValueAveList.append(np.mean(map[segment]))
        segmentValueStdList.append(np.std(map[segment]))


crange=[20,60]

segmentValueAveListPlot=[ new_min + (value - crange[0]) * (new_max - new_min) / (crange[1] - crange[0])
    for value in segmentValueAveList]
output=bullsEye(segmentValueAveListPlot,segmentValueAveList)
'''outputPlot=[ crange[0] + (value - 0) * (crange[1] - crange[0]) / (255 - 0)
    for value in output]'''
%matplotlib qt
#plt.imshow(outputPlot,vmax=crange[1],cmap='magma')
plt.imshow(output,vmax=255,cmap='viridis')
plt.axis('off')
#plt.colorbar()
aha_img_save_path=os.path.join(saveAHA_Path,f'{map_T2.CIRC_ID}_{map_T2.ID}_AHA')
if os.path.exists(saveAHA_Path) is False: 
    os.makedirs(saveAHA_Path)
plt.savefig(aha_img_save_path)
#cv2.imwrite(dirName+'/Bulls_Eye.png', output)
#cv2.imwrite(dirName+'/Stacked_Output.png', stack)
#%%
saveAHA_Path=os.path.join(defaultPath, "saved_ims_v3_June_5_2024","WITH8000","AHA_SingleMap")
new_min=0
new_max=255
segmentValueAveList=[]
segmentValueStdList=[]
segment_16=map_DWI.segment_16
for nn,segment_stack in enumerate(segment_16):
                        #Loop over values in each slice [6, 6, 4]

    for segment in segment_stack:
        #mask=map_T1.mask_lv[:,:,nn].squeeze()
        map=map_DWI._map[:,:,nn].squeeze()
        segmentValueAveList.append(np.mean(map[segment]))
        segmentValueStdList.append(np.std(map[segment]))


crange=[0.5,2.5]

segmentValueAveListPlot=[ new_min + (value - crange[0]) * (new_max - new_min) / (crange[1] - crange[0])
    for value in segmentValueAveList]
output=bullsEye(segmentValueAveListPlot,segmentValueAveList)
'''outputPlot=[ crange[0] + (value - 0) * (crange[1] - crange[0]) / (255 - 0)
    for value in output]'''
%matplotlib qt
#plt.imshow(outputPlot,vmax=crange[1],cmap='magma')
plt.imshow(output,vmax=255,cmap='hot')
plt.axis('off')
#plt.colorbar()
aha_img_save_path=os.path.join(saveAHA_Path,f'{map_DWI.CIRC_ID}_{map_DWI.ID}_AHA')
if os.path.exists(saveAHA_Path) is False: 
    os.makedirs(saveAHA_Path)
plt.savefig(aha_img_save_path)

#%%


#%%




#%%
def imshowMap(obj,path,plot):
    num_slice=obj.Nz
    volume=obj._map
    ID=str('map_' + obj.CIRC_ID + '_' + obj.ID)
    crange=obj.crange
    cmap=obj.cmap
    figsize = (3.4*num_slice, 3)

    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    axes=axes.ravel()
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(volume[..., sl],vmin=crange[0],vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.018, aspect=8)
    img_dir= os.path.join(path,f'{ID}')
    if plot:
        plt.savefig(img_dir)
    pass
#%%
%matplotlib inline
map_T1_post.crange=[0,1600]
img_save_dir=os.path.join(img_root_dir,CIRC_ID)
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir) 
imshowMap(obj=map_T1,plot=plot,path=img_save_dir)
imshowMap(obj=map_T2,plot=plot,path=img_save_dir)
imshowMap(obj=map_DWI,plot=plot,path=img_save_dir)
imshowMap(obj=map_T1_post,plot=plot,path=img_save_dir)

#%%
%matplotlib qt
map_T1.go_crop()
map_T1.go_resize(scale=2)
cropzone=map_T1.cropzone
#%%
map_T2.cropzone=cropzone
map_T2.go_crop()
map_T2.go_resize(scale=2)
map_DWI.cropzone=cropzone
map_DWI.go_crop()
map_DWI.go_resize(scale=2)
#%%
#Crop the map and the data
for map in [map_T1,map_T2,map_DWI]:
    Nz=map.Nz
    data=map._map
    from imgbasics import imcrop
    temp = imcrop(data[:,:,0], cropzone)
    shape = (temp.shape[0], temp.shape[1], Nz)
    data_crop = np.zeros(shape)
    for z in tqdm(range(map.Nz)):
        data_crop[:,:,z]=imcrop(data[:,:,z], cropzone)
    data_crop=imresize(data_crop,np.shape(map._data)[0:3])
    map._map=data_crop.squeeze()
    map._update()
    print(map.shape)

#%%
%matplotlib qt
map_DWI.go_segment_LV(reject=None,z=[0,1,2], image_type="b0_avg",roi_names=['endo', 'epi'])
map_T1._update_mask(map_DWI)
map_T2._update_mask(map_DWI)
map_T2.show_calc_stats_LV()
map_T1.show_calc_stats_LV()
map_DWI.show_calc_stats_LV()
map_T1_post._update_mask(map_DWI)
map_T1_post.show_calc_stats_LV()
#%%
def testing_plot(obj1,obj2,obj3, sl):
    
    %matplotlib inline
    alpha = 1.0*obj1.mask_lv[..., sl]

    print(f"Slice {sl}")
    # map map and overlay
    figsize = (4, 2)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, constrained_layout=True)

    for ind,obj in enumerate([obj1,obj2,obj3]):
        # map
        crange=obj.crange

        axes[ind,0].set_axis_off()
        im = axes[ind,0].imshow(obj._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

        # map overlay
        base_sum=np.array([obj1._data[:, :, sl, 1],obj2._data[:, :, sl, 0],obj3._data[:, :, sl, 0]])
        base_im = np.mean(base_sum,axis=0)
        brightness = 0.8
        axes[ind,1].set_axis_off()
        axes[ind,1].imshow(base_im,vmax=np.max(base_im)*brightness, cmap="gray")
        im = axes[ind,1].imshow(obj._map[..., sl], alpha=alpha, vmin=crange[0], vmax=crange[1], cmap=obj.cmap)

    plt.show()     
    pass
def testing_reseg(obj1,obj2,obj3,plot=plot):
    numberSlice=obj1.Nz
    obj=obj3
    for sl in range(numberSlice):
        testing_plot(obj1,obj2,obj3,sl)
        resegment = True
        
        # while resegment:
        # decide if resegmentation is needed
        print("Perform resegmentation? (Y/N)")
        tmp = input()
        resegment = (tmp == "Y") or (tmp == "y")
        
        while resegment:
            
            print("Resegment endo? (Y/N)")
            tmp = input()
            reseg_endo = (tmp == "Y") or (tmp == "y")
            
            print("Resegment epi? (Y/N)")
            tmp = input()
            reseg_epi = (tmp == "Y") or (tmp == "y") 
            
            roi_names = np.array(["endo", "epi"])
            roi_names = roi_names[np.argwhere([reseg_endo, reseg_epi]).ravel()]
            
            %matplotlib qt  
            # selectively resegment LV
            print("Kernel size: ")
            kernel = int(input())
            obj.go_resegment_LV(z=sl, roi_names=roi_names, dilate=True, kernel=kernel,image_type="b0_aveg")
            
            # re-plot
            testing_plot(obj1,obj2,obj3,sl)

            # resegment?
            print("Perform resegmentation? (Y/N)")
            tmp = input()
            resegment = (tmp == "Y") or (tmp == "y")
            obj1._update_mask(obj)
            obj2._update_mask(obj)
            obj3._update_mask(obj)
            testing_plot(obj1,obj2,obj3,sl)
            obj1.show_calc_stats_LV()
            obj2.show_calc_stats_LV()
            obj3.show_calc_stats_LV()
    if plot:
        obj1.save(filename=os.path.join(img_save_dir,f'{obj1.CIRC_ID}_{obj1.ID}_p.mapping')) 
        obj2.save(filename=os.path.join(img_save_dir,f'{obj2.CIRC_ID}_{obj2.ID}_p.mapping')) 
        obj3.save(filename=os.path.join(img_save_dir,f'{obj3.CIRC_ID}_{obj3.ID}_p.mapping')) 
    pass
#%%
%matplotlib inline
testing_reseg(map_T1,map_T2,map_DWI)
# %% View Maps Overlay

%matplotlib inline
brightness=1.4
# view HAT mask
num_slice = map_T1.Nz 
figsize = (3.4*num_slice, 3)

# T1
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_T1.crange
cmap=map_T1.cmap
base_sum=np.concatenate((map_T2._data[:, :, :, 0:5],map_DWI._data[:, :, :, 0:5]),axis=-1)
base_im = np.mean(base_sum,axis=-1)

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_T1._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_T1.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_T1.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# T2
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_T2.crange
cmap=map_T2.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_T2._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_T2.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_T2.ID}_overlay_maps.png"))
plt.show()  
plt.close()
# ADC
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_DWI.crange
cmap=map_DWI.cmap

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_DWI._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_DWI.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_DWI.ID}_overlay_maps.png"))
plt.show()  


# T1_post
fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
crange=map_T1_post.crange
cmap=map_T1_post.cmap
base_sum=np.concatenate((map_T2._data[:, :, :, 0:5],map_DWI._data[:, :, :, 0:5]),axis=-1)
base_im = np.mean(base_sum,axis=-1)

for sl in range(num_slice):
    axes[sl].set_axis_off()
    axes[sl].imshow(base_im[...,sl], cmap="gray", vmax=np.max(base_im[...,sl]*brightness))
    im = axes[sl].imshow(map_T1_post._map[..., sl], vmin=crange[0], vmax=crange[1], cmap=cmap, alpha=1.0*map_T1_post.mask_lv[...,sl])
#cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=1, pad=0.018, aspect=11)
if plot:
    plt.savefig(os.path.join(img_save_dir, f"{map_T1_post.ID}_overlay_maps.png"))
plt.show()  
plt.close()
#%%
map_T1.show_calc_stats_LV()
map_T2.show_calc_stats_LV()
map_DWI.show_calc_stats_LV()
#%%
MP01.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3000])
MP02.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,150])
MP03.export_stats(filename=r'C:\Research\MRI\MP_EPI\mapping_Jan.csv',crange=[0,3])


# %%
map_T1.save(filename=os.path.join(img_save_dir,f'{map_T1.CIRC_ID}_{map_T1.ID}_p_cropped.mapping'))
map_T2.save(filename=os.path.join(img_save_dir,f'{map_T2.CIRC_ID}_{map_T2.ID}_p_cropped.mapping'))
map_DWI.save(filename=os.path.join(img_save_dir,f'{map_DWI.CIRC_ID}_{map_DWI.ID}_p_cropped.mapping'))

# %%
