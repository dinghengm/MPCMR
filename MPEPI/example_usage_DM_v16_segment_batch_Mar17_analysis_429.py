###############From contours to analysis
##############From  saved_ims_v2_Feb_5_2024/NULL/xxxp.pickle
##############To: misregistered area, area, and sharpness
# #############The data inside are masklv, mask_endo, mask_epi
#############Shape Nx,Ny,Nd
################
##############
###############This file can input xxxp.pickle
###############Then from data/contours generate AREA, MP01, MP02,MP03, All, Epi, Endo, 

#%%
from libMapping_v13 import mapping  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
defaultPath= r'C:\Research\MRI\MP_EPI'
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.rcParams.update({'axes.titlesize': 'small'})
import matplotlib
from roipoly import RoiPoly, MultiRoi
from skimage.transform import resize as imresize
from imgbasics import imcrop #for croping
import pickle
from skimage import measure
matplotlib.rcParams['savefig.dpi'] = 400
plot=True

resolution=1.4
#%%
def exportData(data,filename):
        if os.path.isfile(filename):    
            data.to_csv(filename, index=False, header=False, mode='a')
        else:
            data.to_csv(filename, index=False)

        print('Saved '+ filename +' successfully!')
def calc_misArea(mask,ind_label,slice,label,resolution=1.4):
    '''
        #Calculate the mis are between target and prediction
        #A and B - A
        Input: mask
            ind_label [list]: 
                list[0] -> end of MP01
                list[1] -> end of MP02
            slice= slice number 0/1/2
            label= 'endo' or 'epi'
        return:
            misarea: dict with MP01,MP02,MP03,gobal mean, gobal 
    ''' 
    #Store the MP01,MP02,MP03mis area
    valList=[]
    valDict={}
    ind=ind_label[1]   #The image of MP02 (target)
    target=mask[:,:,ind_label[1]]
    for contrast in range(3):
        #Store the list ind in MP01/02/03 with mask
        indList_tmp=[]
        #Store the area of each repetition 
        areaList=[]
        for i in range(ind_label[contrast],ind_label[contrast+1]):
            if np.sum(mask[:,:,i])==0:
                continue
            else:
                indList_tmp.append(i)
        print(f'The number of repetition is',len(indList_tmp))
        print(indList_tmp)
        for dd in indList_tmp:
            
            if dd==ind:
                #If target, remove
                continue
            prediction=mask[:,:,dd]
            intersection = np.logical_and(target, prediction)

            misregstered=intersection^target
            subpixel=np.sum(misregstered==1)
            area=subpixel*resolution*resolution        #1.4*1.4mm^2
            areaList.append(area)
        mis_area=np.sum(areaList)/len(areaList)
        #print(mis_area)    
        valDict[rf'Slice{slice}_{label}_MP0{contrast+1}']=np.round(mis_area,2)
        try:
            valList.append(mis_area)
        except:
            continue
    valDict[rf'Slice{slice}_{label}_global']=np.round(np.mean(valList),2)
    valDict[rf'Slice{slice}_{label}_global std']=np.round(np.std(valList),2)
    return valDict

def calc_misArea_perArea(mask,ind_label,slice,label,resolution=1.4):
    '''
        #Calculate the mis are between target and prediction
        #A and B - A
        Input: mask
            ind_label [list]: 
                list[0] -> end of MP01
                list[1] -> end of MP02
            slice= slice number 0/1/2
            label= 'endo' or 'epi'
        return:
            misarea: dict with MP01,MP02,MP03,gobal mean, gobal 

    ''' 
    

    #Store the MP01,MP02,MP03mis area
    valList=[]
    valDict={}
    ind=ind_label[1]   #The image of MP02 (target)
    try:
        target=mask[:,:,ind]
        targetUnitArea=np.sum(target==1)
        while np.sum(target)==0:
            ind+=1
            target=mask[:,:,ind]
            targetUnitArea=np.sum(target==1)
    except:
        pass
    for contrast in range(3):
        #Store the list ind in MP01/02/03 with mask
        indList_tmp=[]
        #Store the area of each repetition 
        areaList=[]
        for i in range(ind_label[contrast],ind_label[contrast+1]):
            if np.sum(mask[:,:,i])==0:
                continue
            else:
                indList_tmp.append(i)
        print('The number of repetition is',len(indList_tmp))
        print(indList_tmp)
        for dd in indList_tmp:
            
            if dd==ind:
                #If target remove
                continue
            prediction=mask[:,:,dd]
            intersection = np.logical_and(target, prediction)

            misregstered=intersection^target
            subpixel=np.sum(misregstered==1)
            unitSubpixel=subpixel/targetUnitArea
            unitarea=unitSubpixel*resolution*resolution        #1.4*1.4mm^2
            areaList.append(unitarea)
        mis_area_unit=np.sum(areaList)/len(areaList)
        #print(mis_area)    
        valDict[f'Slice{slice}_{label}_MP0{contrast+1}']=np.round(mis_area_unit,2)
        try:
            valList.append(mis_area_unit)
        except:
            continue
    valDict[f'Slice{slice}_{label}_global']=np.round(np.mean(valList),2)
    valDict[f'Slice{slice}_{label}_global std']=np.round(np.std(valList),2)
    return valDict




#%%
####################405 has to regenerate MP03-DWI######################
CIRC_ID_List=[446,429,419,398,382,381,373,472,486,498,500]
#CIRC_NUMBER=CIRC_ID_List[9]

CIRC_NUMBER=CIRC_ID_List[1]
CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
print(f'Running{CIRC_ID}')
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','NULL',f'{CIRC_ID}')
img_save_dir=img_root_dir
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)
MP01_0=mapping(mapList[0])
MP01_1=mapping(mapList[1])
MP01_2=mapping(mapList[2])
img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
mapList=[]
for dirpath,dirs,files in  os.walk(img_root_dir):
    for x in files:
        path=os.path.join(dirpath,x)
        if path.endswith('m.mapping'):
            mapList.append(path)
MP01=mapping(mapList[3])
MP02=mapping(mapList[4])
#dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
#MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
MP03=mapping(mapList[5])
print(mapList)
#%%
#Read the ROI from all npy array

#Def misregistered area/area
#Test load
#dataDict is the npy array
dataDict = {}
for ss in range(3):
    dataDict[f'Slice{ss}_data']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data.npy'))
    dataDict[f'Slice{ss}_endo']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo.npy'))
    dataDict[f'Slice{ss}_epi']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi.npy'))
    dataDict[f'Slice{ss}_lv']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv.npy'))
    dataDict[f'Slice{ss}_data_raw']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_data_raw.npy'))
    dataDict[f'Slice{ss}_endo_raw']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_endo_raw.npy'))
    dataDict[f'Slice{ss}_epi_raw']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_epi_raw.npy'))
    dataDict[f'Slice{ss}_lv_raw']=np.load(os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_Slice{ss}_masklv_raw.npy'))


#%%
MP01_list=[MP01_0,MP01_1,MP01_2]
dictData={}
dictData_raw={}
#moco
dictData['CIRC_ID']=CIRC_ID
dictData['ID']='moco'
dictData_raw['CIRC_ID']=CIRC_ID
dictData_raw['ID']='raw'
for ss,mp01 in enumerate(MP01_list):
    #Moco

    ind_label=[]
    indtmp=np.shape(mp01._data)[-1]    
    ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
    mask_endo=dataDict[f'Slice{ss}_endo']
    dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
    dictData.update(dicttmp)
    #epi
    mask_epi=dataDict[f'Slice{ss}_epi']
    dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
    dictData.update(dicttmp)
    #lv
    mask_lv=dataDict[f'Slice{ss}_lv']
    dicttmp=calc_misArea(mask_lv,ind_label,ss,'lv')
    dictData.update(dicttmp)
    #Raw
    ind_label=[]
    indtmp=np.shape(mp01._raw_data)[-1]    
    ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
    mask_endo=dataDict[f'Slice{ss}_endo_raw']
    dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
    dictData_raw.update(dicttmp)
    mask_epi=dataDict[f'Slice{ss}_epi_raw']
    dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
    dictData_raw.update(dicttmp)
    #raw
    mask_lv=dataDict[f'Slice{ss}_lv_raw']
    dicttmp=calc_misArea(mask_lv,ind_label,ss,'lv')
    dictData_raw.update(dicttmp)



#%%
#########Save the dict


moco_stats = pd.DataFrame(dictData, index=[0])
moco_raw_stats = pd.DataFrame(dictData_raw, index=[0])

#%%
#Calculate the mean of MP01,MP02,MP03??
# Global_MP01_endo	
# Global_MP02_endo	
# Global_MP03_endo	
# Global_MP01_epi	
# Global_MP02_epi	
# Global_MP03_epi	
# Global_enco	
# Global_epi
for df in [moco_stats,moco_raw_stats]:
    for area in ['endo','epi','lv']:
        for contrast in [1,2,3]:
            columns_to_average = [f'Slice0_{area}_MP0{contrast}', f'Slice1_{area}_MP0{contrast}', f'Slice2_{area}_MP0{contrast}']
            df[f'Global_MP0{contrast}_{area}']=df[columns_to_average].mean().mean()

for df in [moco_stats,moco_raw_stats]:
    for area in ['endo','epi','lv']:
        columns_to_average = [f'Global_MP01_{area}', f'Global_MP02_{area}', f'Global_MP03_{area}']
        df[f'Global_{area}']=df[columns_to_average].mean().mean()


#%%

###############As of March 15 old version
#filename=os.path.join(defaultPath,f'MisRegristeredArea.csv')
##########Modification on March 15
filename=os.path.join(defaultPath,f'MisRegristeredArea_March15.csv')
def exportData(data,filename):
    if os.path.isfile(filename):    
        data.to_csv(filename, index=False, header=False, mode='a')
    else:
        data.to_csv(filename, index=False)

    print('Saved '+ filename +' successfully!')
exportData(moco_stats,filename)
exportData(moco_raw_stats,filename)

#%%
#########Normalized by unit area
MP01_list=[MP01_0,MP01_1,MP01_2]
dictData={}
dictData_raw={}
#moco
dictData['CIRC_ID']=CIRC_ID
dictData['ID']='moco'
dictData_raw['CIRC_ID']=CIRC_ID
dictData_raw['ID']='raw'
for ss,mp01 in enumerate(MP01_list):
    #Moco

    ind_label=[]
    indtmp=np.shape(mp01._data)[-1]    
    ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
    mask_endo=dataDict[f'Slice{ss}_endo']
    dicttmp=calc_misArea_perArea(mask_endo,ind_label,ss,'endo')
    dictData.update(dicttmp)
    mask_epi=dataDict[f'Slice{ss}_epi']
    dicttmp=calc_misArea_perArea(mask_epi,ind_label,ss,'epi')
    dictData.update(dicttmp)
    #Raw
    ind_label=[]
    indtmp=np.shape(mp01._raw_data)[-1]    
    ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
    mask_endo=dataDict[f'Slice{ss}_endo_raw']
    dicttmp=calc_misArea_perArea(mask_endo,ind_label,ss,'endo')
    dictData_raw.update(dicttmp)
    mask_epi=dataDict[f'Slice{ss}_epi_raw']
    dicttmp=calc_misArea_perArea(mask_epi,ind_label,ss,'epi')
    dictData_raw.update(dicttmp)
#%%
#########Save the dict


moco_stats = pd.DataFrame(dictData, index=[0])
moco_raw_stats = pd.DataFrame(dictData_raw, index=[0])

filename=os.path.join(defaultPath,f'MisRegristeredAreaperUnit_March15.csv')
def exportData(data,filename):
    if os.path.isfile(filename):    
        data.to_csv(filename, index=False, header=False, mode='a')
    else:
        data.to_csv(filename, index=False)

    print('Saved '+ filename +' successfully!')
exportData(moco_stats,filename)
exportData(moco_raw_stats,filename)


#%%
#bmode
def bmode(data=None,ID=None,x=None,y=None,plot=False,path=None):

    Nx,Ny,Nz,Nd=np.shape(data)
    if x==None and y==None:
        if np.shape(data)[0]>np.shape(data)[1]:
            x=int(np.shape(data)[0]/2)
        else:
            y=int(np.shape(data)[1]/2)
    if x is not None:
        A2=np.zeros((Ny,Nz,Nd),dtype=np.float64)
        A2=data[x,:,:,:]
    elif y is not None:
        A2=np.zeros((Nx,Nz,Nd),dtype=np.float64)
        A2=data[:,y,:,:]
    if Nz !=1:
        fig,axs=plt.subplots(2,Nz)
        ax=axs.ravel()
        for i in range(Nz):
            ax[1,i].imshow(data[...,i,0],cmap='gray')
            if x is not None:
                ax[1,i].axhline(y=x, color='r', linestyle='-')
            if y is not None:
                ax[1,i].axvline(x=y, color='r', linestyle='-')
            ax[2,i].imshow(A2[...,i,:],cmap='gray')
            ax[2,i].set_title(f'z={i}')
            ax[1,i].axis('off')
            ax[2,i].axis('off')
    elif Nz==1:
        plt.subplot(121)
        plt.imshow(data.squeeze()[...,0],cmap='gray')
        plt.axis('off')
        if x is not None:
            plt.axhline(y=x, color='r', linestyle='-')
        if y is not None:
            plt.axvline(x=y, color='r', linestyle='-')
        plt.subplot(122)
        A3=np.squeeze(A2)
        plt.imshow(A3,cmap='gray')
        plt.axis('off')
    if plot==True:
        dir=os.path.join(path,ID)
        plt.savefig(dir)
    plt.show()
for ss in range(3):
    print('raw')
    plt.figure(constrained_layout=True)
    bmode(data=dataDict[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],
            plot=True,
            ID=f'Slice{ss}_bmode_raw',
            path=img_save_dir)
    print('moco')
    plt.figure(constrained_layout=True)
    bmode(data=dataDict[f'Slice{ss}_data'][:,:,np.newaxis,:],
            plot=True,
            ID=f'Slice{ss}_bmode',
            path=img_save_dir)
#MP02.bmode(data=dataDict1[f'Slice{ss}_data'][:,:,np.newaxis,:])
#%%
def plotAve(data,masklv,plot=False,path=None,ID=None):
    data=data.squeeze()
    masklv=masklv.squeeze()
    indList=[]
    for i in range(np.shape(masklv)[-1]):
        if np.sum(masklv[:,:,i])==0:
            continue
        else:
            indList.append(i)
    Nx,Ny,Nd=np.shape(data.squeeze())
    dataNew=np.zeros((Nx,Ny))
    dataNew = np.mean(data[...,indList],axis=-1)
    plt.imshow(dataNew,cmap='gray')
    plt.axis('off')
    if plot==True:
        dir=os.path.join(path,ID)
        plt.savefig(dir)

    plt.show()
for ss in range(3):
    print('raw')
    plotAve(data=dataDict[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],
            masklv=dataDict[f'Slice{ss}_lv_raw'][:,:,np.newaxis,:],
            plot=True,
            ID=f'Slice{ss}_mean_raw',
            path=img_save_dir)
    print('moco')
    plotAve(data=dataDict[f'Slice{ss}_data'][:,:,np.newaxis,:],
            masklv=dataDict[f'Slice{ss}_lv'][:,:,np.newaxis,:], 
            plot=True,
            ID=f'Slice{ss}_mean',
            path=img_save_dir)




#%%

%matplotlib inline
plt.figure(constrained_layout=True)
for ss in range(3):
    plt.subplot(3,2,2*ss+1)

    image_data=dataDict[f'Slice{ss}_data']
    mask_lv=dataDict[f'Slice{ss}_lv']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')

for ss in range(3):
    plt.subplot(3,2,2*ss+2)
    image_data=dataDict[f'Slice{ss}_data_raw']
    mask_lv=dataDict[f'Slice{ss}_lv_raw']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')
plt.savefig(os.path.join(img_save_dir,f'contours'))
plt.show()

#%%
##############Plot contours based on MP010203
%matplotlib inline
plt.figure(constrained_layout=True)
MP01_list=[MP01_0,MP01_1,MP01_2]
for ss in range(3):
    

    image_data=dataDict[f'Slice{ss}_data']
    mask_lv=dataDict[f'Slice{ss}_lv']
    mp01=MP01_list[ss]
    indtmp=np.shape(mp01._data)[-1]   
    ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]

    #MP01
    for contrast in range(3):
        plt.subplot(3,4,4*ss+contrast+2)
        plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
        indList_tmp=range(ind_label[contrast],ind_label[contrast+1])
        for d in indList_tmp:
            image=mask_lv[:,:,d]*255-1
            img=np.float16(image)
            contours = measure.find_contours(img, 240)
            try:
                plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
                plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
            except:
                continue
            plt.axis('off')

for ss in range(3):
    plt.subplot(3,4,4*ss+1)
    image_data=dataDict[f'Slice{ss}_data_raw']
    mask_lv=dataDict[f'Slice{ss}_lv_raw']
    Nd=np.shape(image_data)[-1]
    plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
    for d in range(Nd):
        image=mask_lv[:,:,d]*255-1
        img=np.float16(image)

        contours = measure.find_contours(img, 240)
        try:
            plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
            plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
        except:
            continue
        plt.axis('off')
filename_dir=os.path.join(img_save_dir,'BasedonMP')
if not os.path.exists(filename_dir):
    os.mkdir(filename_dir)
plt.savefig(os.path.join(filename_dir,f'contours_BasedonMP'))
plt.show()


#%%
#Plot ave based on MP01,MP02,MP03: raw at first and the other at the ends
plt.figure(constrained_layout=True)
MP01_list=[MP01_0,MP01_1,MP01_2]
for ss in range(3):
    

    image_data=dataDict[f'Slice{ss}_data']
    mask_lv=dataDict[f'Slice{ss}_lv']
    mp01=MP01_list[ss]
    indtmp=np.shape(mp01._data)[-1]   
    ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]

    #MP01
    for contrast in range(3):
        plt.subplot(3,4,4*ss+contrast+2)
        indList_tmp=range(ind_label[contrast],ind_label[contrast+1])
        #The 
        indList=[]
        for d in indList_tmp:
            if np.sum(mask_lv[:,:,d])==0:
                continue
            else:
                indList.append(d)
        Nx,Ny,Nd=np.shape(image_data.squeeze())
        dataNew=np.zeros((Nx,Ny))
        dataNew = np.mean(image_data[...,indList],axis=-1)
        try:
            plt.imshow(dataNew,cmap='gray')
        except:
            continue
        plt.axis('off')
    

for ss in range(3):
    plt.subplot(3,4,4*ss+1)
    image_data=dataDict[f'Slice{ss}_data_raw']
    mask_lv=dataDict[f'Slice{ss}_lv_raw']
    Nd=np.shape(image_data)[-1]
    indList=[]
    for d in range(Nd):
        if np.sum(mask_lv[:,:,d])==0:
                continue
        else:
            indList.append(d)

    Nx,Ny,Nd=np.shape(image_data.squeeze())
    dataNew=np.zeros((Nx,Ny))
    dataNew = np.mean(image_data[...,indList],axis=-1)
    try:
        plt.imshow(dataNew,cmap='gray')
    except:
        continue
    plt.axis('off')
    
filename_dir=os.path.join(img_save_dir,'BasedonMP')
if not os.path.exists(filename_dir):
    os.mkdir(filename_dir)
plt.savefig(os.path.join(filename_dir,f'Average_BasedonMP'))
plt.show()





#%%
##Iterate #Same as before.
iterate=[0,1,2,4]
for i in iterate:
    CIRC_NUMBER=CIRC_ID_List[i]
    CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
    print(f'Running{CIRC_ID}')
    img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','NULL',f'{CIRC_ID}')
    img_save_dir=img_root_dir
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('m.mapping'):
                mapList.append(path)
    MP01_0=mapping(mapList[0])
    MP01_1=mapping(mapList[1])
    MP01_2=mapping(mapList[2])
    img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
    mapList=[]
    for dirpath,dirs,files in  os.walk(img_root_dir):
        for x in files:
            path=os.path.join(dirpath,x)
            if path.endswith('m.mapping'):
                mapList.append(path)
    MP01=mapping(mapList[3])
    MP02=mapping(mapList[4])
    #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
    #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
    MP03=mapping(mapList[5])
    print(mapList)

    #Def misregistered area/area
    #Test load
    saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
    with open(saveDict_name, 'rb') as inp:
        dataDict = pickle.load(inp)
    print(saveDict_name)
    print('dictionary Load successfully to file')


    MP01_list=[MP01_0,MP01_1,MP01_2]
    dictData={}
    dictData_raw={}
    #moco
    dictData['CIRC_ID']=CIRC_ID
    dictData['ID']='moco'
    dictData_raw['CIRC_ID']=CIRC_ID
    dictData_raw['ID']='raw'
    for ss,mp01 in enumerate(MP01_list):
        #Moco

        ind_label=[]
        indtmp=np.shape(mp01._data)[-1]    
        ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
        mask_endo=dataDict[f'Slice{ss}_endo']
        dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
        dictData.update(dicttmp)
        mask_epi=dataDict[f'Slice{ss}_epi']
        dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
        dictData.update(dicttmp)
        #Raw
        ind_label=[]
        indtmp=np.shape(mp01._raw_data)[-1]    
        ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
        mask_endo=dataDict[f'Slice{ss}_endo_raw']
        dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
        dictData_raw.update(dicttmp)
        mask_epi=dataDict[f'Slice{ss}_epi_raw']
        dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
        dictData_raw.update(dicttmp)
    #########Save the dict
    moco_stats = pd.DataFrame(dictData, index=[0])
    moco_raw_stats = pd.DataFrame(dictData_raw, index=[0])
    filename=os.path.join(defaultPath,f'MisRegristeredArea.csv')
    
    exportData(moco_stats,filename)
    exportData(moco_raw_stats,filename)


#%%
'''LV sharpness of the mean image across all DWIs was
compared for conventional MOCO, MT-MOCO, and breath-
hold reconstructions defined by the inverse of the distance
of the 20% to 80% of the maximum signal intensity at the
lateral and septal walls of the LV along a line defined by the
midpoint of each wall and the center of mass of the blood
pool.
As in https://pubs.rsna.org/doi/epdf/10.1148/radiology.219.1.r01ap37270
'''
def cal_sharpness(data,masklv,x=None,y=None,resolution=1.4):
    data=data.squeeze()
    masklv=masklv.squeeze()
    if x==None and y==None:
        x=int(np.shape(data)[0]/2)
    #Loop with make on it
    indList=[]
    for i in range(np.shape(masklv)[-1]):
        if np.sum(masklv[:,:,i])==0:
            continue
        else:
            indList.append(i)
    #Loop for Nd
    for dd in indList:
        image=data[...,dd]
        mask_image=image*masklv[...,0]
        if x is not None:
            line=mask_image[x,...]
        elif y is not None:
            line=mask_image[:,y]
        ind=np.where(np.logical_and(line>=np.max(line)*0.2, line<=np.max(line)*0.8))

        #FROM 20-80%
        #    



#%%
###For loop
for ind in range(len(CIRC_ID_List)):
    try:
        CIRC_NUMBER=CIRC_ID_List[ind]
        CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
        print(f'Running{CIRC_ID}')
        img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','NULL',f'{CIRC_ID}')
        img_save_dir=img_root_dir
        mapList=[]
        for dirpath,dirs,files in  os.walk(img_root_dir):
            for x in files:
                path=os.path.join(dirpath,x)
                if path.endswith('m.mapping'):
                    mapList.append(path)
        MP01_0=mapping(mapList[0])
        MP01_1=mapping(mapList[1])
        MP01_2=mapping(mapList[2])
        img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
        mapList=[]
        for dirpath,dirs,files in  os.walk(img_root_dir):
            for x in files:
                path=os.path.join(dirpath,x)
                if path.endswith('m.mapping'):
                    mapList.append(path)
        MP01=mapping(mapList[3])
        MP02=mapping(mapList[4])
        #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
        #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
        MP03=mapping(mapList[5])
        print(mapList)

        #Def misregistered area/area
        #Test load
        saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
        with open(saveDict_name, 'rb') as inp:
            dataDict = pickle.load(inp)
        print(saveDict_name)
        print('dictionary Load successfully to file')




        MP01_list=[MP01_0,MP01_1,MP01_2]
        dictData={}
        dictData_raw={}
        #moco
        dictData['CIRC_ID']=CIRC_ID
        dictData['ID']='moco'
        dictData_raw['CIRC_ID']=CIRC_ID
        dictData_raw['ID']='raw'
        for ss,mp01 in enumerate(MP01_list):
            #Moco

            ind_label=[]
            indtmp=np.shape(mp01._data)[-1]    
            ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
            mask_endo=dataDict[f'Slice{ss}_endo']
            dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
            dictData.update(dicttmp)
            mask_epi=dataDict[f'Slice{ss}_epi']
            dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
            dictData.update(dicttmp)
            #Raw
            ind_label=[]
            indtmp=np.shape(mp01._raw_data)[-1]    
            ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
            mask_endo=dataDict[f'Slice{ss}_endo_raw']
            dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
            dictData_raw.update(dicttmp)
            mask_epi=dataDict[f'Slice{ss}_epi_raw']
            dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
            dictData_raw.update(dicttmp)
        #########Save the dict


        moco_stats = pd.DataFrame(dictData, index=[0])
        moco_raw_stats = pd.DataFrame(dictData_raw, index=[0])







        filename=os.path.join(defaultPath,f'MisRegristeredArea.csv')
        def exportData(data,filename):
            if os.path.isfile(filename):    
                data.to_csv(filename, index=False, header=False, mode='a')
            else:
                data.to_csv(filename, index=False)

            print('Saved '+ filename +' successfully!')
        exportData(moco_stats,filename)
        exportData(moco_raw_stats,filename)

        #########Normalized by unit area
        MP01_list=[MP01_0,MP01_1,MP01_2]
        dictData={}
        dictData_raw={}
        #moco
        dictData['CIRC_ID']=CIRC_ID
        dictData['ID']='moco'
        dictData_raw['CIRC_ID']=CIRC_ID
        dictData_raw['ID']='raw'
        for ss,mp01 in enumerate(MP01_list):
            #Moco

            ind_label=[]
            indtmp=np.shape(mp01._data)[-1]    
            ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
            mask_endo=dataDict[f'Slice{ss}_endo']
            dicttmp=calc_misArea_perArea(mask_endo,ind_label,ss,'endo')
            dictData.update(dicttmp)
            mask_epi=dataDict[f'Slice{ss}_epi']
            dicttmp=calc_misArea_perArea(mask_epi,ind_label,ss,'epi')
            dictData.update(dicttmp)
            #Raw
            ind_label=[]
            indtmp=np.shape(mp01._raw_data)[-1]    
            ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
            mask_endo=dataDict[f'Slice{ss}_endo_raw']
            dicttmp=calc_misArea_perArea(mask_endo,ind_label,ss,'endo')
            dictData_raw.update(dicttmp)
            mask_epi=dataDict[f'Slice{ss}_epi_raw']
            dicttmp=calc_misArea_perArea(mask_epi,ind_label,ss,'epi')
            dictData_raw.update(dicttmp)
        #########Save the dict


        moco_stats = pd.DataFrame(dictData, index=[0])
        moco_raw_stats = pd.DataFrame(dictData_raw, index=[0])

        filename=os.path.join(defaultPath,f'MisRegristeredAreaperUnit.csv')
        def exportData(data,filename):
            if os.path.isfile(filename):    
                data.to_csv(filename, index=False, header=False, mode='a')
            else:
                data.to_csv(filename, index=False)

            print('Saved '+ filename +' successfully!')
        exportData(moco_stats,filename)
        exportData(moco_raw_stats,filename)


        #bmode
        def bmode(data=None,ID=None,x=None,y=None,plot=False,path=None):

            Nx,Ny,Nz,Nd=np.shape(data)
            if x==None and y==None:
                if np.shape(data)[0]>np.shape(data)[1]:
                    x=int(np.shape(data)[0]/2)
                else:
                    y=int(np.shape(data)[1]/2)
            if x is not None:
                A2=np.zeros((Ny,Nz,Nd),dtype=np.float64)
                A2=data[x,:,:,:]
            elif y is not None:
                A2=np.zeros((Nx,Nz,Nd),dtype=np.float64)
                A2=data[:,y,:,:]
            if Nz !=1:
                fig,axs=plt.subplots(2,Nz)
                ax=axs.ravel()
                for i in range(Nz):
                    ax[1,i].imshow(data[...,i,0],cmap='gray')
                    if x is not None:
                        ax[1,i].axhline(y=x, color='r', linestyle='-')
                    if y is not None:
                        ax[1,i].axvline(x=y, color='r', linestyle='-')
                    ax[2,i].imshow(A2[...,i,:],cmap='gray')
                    ax[2,i].set_title(f'z={i}')
                    ax[1,i].axis('off')
                    ax[2,i].axis('off')
            elif Nz==1:
                plt.subplot(121)
                plt.imshow(data.squeeze()[...,0],cmap='gray')
                plt.axis('off')
                if x is not None:
                    plt.axhline(y=x, color='r', linestyle='-')
                if y is not None:
                    plt.axvline(x=y, color='r', linestyle='-')
                plt.subplot(122)
                A3=np.squeeze(A2)
                plt.imshow(A3,cmap='gray')
                plt.axis('off')
            if plot==True:
                dir=os.path.join(path,ID)
                plt.savefig(dir)
            plt.show()
        for ss in range(3):
            print('raw')
            plt.figure(constrained_layout=True)
            bmode(data=dataDict[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],
                    plot=True,
                    ID=f'Slice{ss}_bmode_raw',
                    path=img_save_dir)
            print('moco')
            plt.figure(constrained_layout=True)
            bmode(data=dataDict[f'Slice{ss}_data'][:,:,np.newaxis,:],
                    plot=True,
                    ID=f'Slice{ss}_bmode',
                    path=img_save_dir)
        #MP02.bmode(data=dataDict1[f'Slice{ss}_data'][:,:,np.newaxis,:])

        def plotAve(data,masklv,plot=False,path=None,ID=None):
            data=data.squeeze()
            masklv=masklv.squeeze()
            indList=[]
            for i in range(np.shape(masklv)[-1]):
                if np.sum(masklv[:,:,i])==0:
                    continue
                else:
                    indList.append(i)
            Nx,Ny,Nd=np.shape(data.squeeze())
            dataNew=np.zeros((Nx,Ny))
            dataNew = np.mean(data[...,indList],axis=-1)
            plt.imshow(dataNew,cmap='gray')
            plt.axis('off')
            if plot==True:
                dir=os.path.join(path,ID)
                plt.savefig(dir)

            plt.show()
        for ss in range(3):
            print('raw')
            plotAve(data=dataDict[f'Slice{ss}_data_raw'][:,:,np.newaxis,:],
                    masklv=dataDict[f'Slice{ss}_lv_raw'][:,:,np.newaxis,:],
                    plot=True,
                    ID=f'Slice{ss}_mean_raw',
                    path=img_save_dir)
            print('moco')
            plotAve(data=dataDict[f'Slice{ss}_data'][:,:,np.newaxis,:],
                    masklv=dataDict[f'Slice{ss}_lv'][:,:,np.newaxis,:], 
                    plot=True,
                    ID=f'Slice{ss}_mean',
                    path=img_save_dir)



        %matplotlib inline
        plt.figure(constrained_layout=True)
        for ss in range(3):
            plt.subplot(3,2,2*ss+1)

            image_data=dataDict[f'Slice{ss}_data']
            mask_lv=dataDict[f'Slice{ss}_lv']
            Nd=np.shape(image_data)[-1]
            plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
            for d in range(Nd):
                image=mask_lv[:,:,d]*255-1
                img=np.float16(image)

                contours = measure.find_contours(img, 240)
                try:
                    plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
                    plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
                except:
                    continue
                plt.axis('off')

        for ss in range(3):
            plt.subplot(3,2,2*ss+2)
            image_data=dataDict[f'Slice{ss}_data_raw']
            mask_lv=dataDict[f'Slice{ss}_lv_raw']
            Nd=np.shape(image_data)[-1]
            plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
            for d in range(Nd):
                image=mask_lv[:,:,d]*255-1
                img=np.float16(image)

                contours = measure.find_contours(img, 240)
                try:
                    plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
                    plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
                except:
                    continue
                plt.axis('off')
        plt.savefig(os.path.join(img_save_dir,f'contours'))
        plt.show()

        ##############Plot contours based on MP010203
        %matplotlib inline
        plt.figure(constrained_layout=True)
        MP01_list=[MP01_0,MP01_1,MP01_2]
        for ss in range(3):
            

            image_data=dataDict[f'Slice{ss}_data']
            mask_lv=dataDict[f'Slice{ss}_lv']
            mp01=MP01_list[ss]
            indtmp=np.shape(mp01._data)[-1]   
            ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]

            #MP01
            for contrast in range(3):
                plt.subplot(3,4,4*ss+contrast+2)
                plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
                indList_tmp=range(ind_label[contrast],ind_label[contrast+1])
                for d in indList_tmp:
                    image=mask_lv[:,:,d]*255-1
                    img=np.float16(image)
                    contours = measure.find_contours(img, 240)
                    try:
                        plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
                        plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
                    except:
                        continue
                    plt.axis('off')

        for ss in range(3):
            plt.subplot(3,4,4*ss+1)
            image_data=dataDict[f'Slice{ss}_data_raw']
            mask_lv=dataDict[f'Slice{ss}_lv_raw']
            Nd=np.shape(image_data)[-1]
            plt.imshow(image_data[...,0].squeeze(),cmap='gray',vmax=1e10)
            for d in range(Nd):
                image=mask_lv[:,:,d]*255-1
                img=np.float16(image)

                contours = measure.find_contours(img, 240)
                try:
                    plt.plot(contours[0][:, 1], contours[0][:, 0] ,"r", linewidth=0.7)
                    plt.plot(contours[1][:, 1], contours[1][:, 0] ,"g", linewidth=0.7)
                except:
                    continue
                plt.axis('off')
        filename_dir=os.path.join(img_save_dir,'BasedonMP')
        if not os.path.exists(filename_dir):
            os.mkdir(filename_dir)
        plt.savefig(os.path.join(filename_dir,f'contours_BasedonMP'))
        plt.show()

        #Plot ave based on MP01,MP02,MP03: raw at first and the other at the ends
        plt.figure(constrained_layout=True)
        MP01_list=[MP01_0,MP01_1,MP01_2]
        for ss in range(3):
            

            image_data=dataDict[f'Slice{ss}_data']
            mask_lv=dataDict[f'Slice{ss}_lv']
            mp01=MP01_list[ss]
            indtmp=np.shape(mp01._data)[-1]   
            ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]

            #MP01
            for contrast in range(3):
                plt.subplot(3,4,4*ss+contrast+2)
                indList_tmp=range(ind_label[contrast],ind_label[contrast+1])
                #The 
                indList=[]
                for d in indList_tmp:
                    if np.sum(mask_lv[:,:,d])==0:
                        continue
                    else:
                        indList.append(d)
                Nx,Ny,Nd=np.shape(image_data.squeeze())
                dataNew=np.zeros((Nx,Ny))
                dataNew = np.mean(image_data[...,indList],axis=-1)
                try:
                    plt.imshow(dataNew,cmap='gray')
                except:
                    continue
                plt.axis('off')
            

        for ss in range(3):
            plt.subplot(3,4,4*ss+1)
            image_data=dataDict[f'Slice{ss}_data_raw']
            mask_lv=dataDict[f'Slice{ss}_lv_raw']
            Nd=np.shape(image_data)[-1]
            indList=[]
            for d in range(Nd):
                if np.sum(mask_lv[:,:,d])==0:
                        continue
                else:
                    indList.append(d)

            Nx,Ny,Nd=np.shape(image_data.squeeze())
            dataNew=np.zeros((Nx,Ny))
            dataNew = np.mean(image_data[...,indList],axis=-1)
            try:
                plt.imshow(dataNew,cmap='gray')
            except:
                continue
            plt.axis('off')
            
        filename_dir=os.path.join(img_save_dir,'BasedonMP')
        if not os.path.exists(filename_dir):
            os.mkdir(filename_dir)
        plt.savefig(os.path.join(filename_dir,f'Average_BasedonMP'))
        plt.show()


        ##Iterate #Same as before.
        iterate=[0,1,2,4]
        for i in iterate:
            CIRC_NUMBER=CIRC_ID_List[i]
            CIRC_ID=f'CIRC_00{CIRC_NUMBER}'
            print(f'Running{CIRC_ID}')
            img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','NULL',f'{CIRC_ID}')
            img_save_dir=img_root_dir
            mapList=[]
            for dirpath,dirs,files in  os.walk(img_root_dir):
                for x in files:
                    path=os.path.join(dirpath,x)
                    if path.endswith('m.mapping'):
                        mapList.append(path)
            MP01_0=mapping(mapList[0])
            MP01_1=mapping(mapList[1])
            MP01_2=mapping(mapList[2])
            img_root_dir=os.path.join(defaultPath,'saved_ims_v2_Feb_5_2024','WITH8000',f'{CIRC_ID}')
            mapList=[]
            for dirpath,dirs,files in  os.walk(img_root_dir):
                for x in files:
                    path=os.path.join(dirpath,x)
                    if path.endswith('m.mapping'):
                        mapList.append(path)
            MP01=mapping(mapList[3])
            MP02=mapping(mapList[4])
            #dicomPath=os.path.join(defaultPath,f'{CIRC_ID}_22737_{CIRC_ID}_22737\MP03_DWI')
            #MP03 = mapping(data=dicomPath,CIRC_ID=CIRC_ID,reject=False,bFilenameSorted=False)
            MP03=mapping(mapList[5])
            print(mapList)

            #Def misregistered area/area
            #Test load
            saveDict_name=os.path.join(img_save_dir,f'{MP01_0.CIRC_ID}_data.pkl')
            with open(saveDict_name, 'rb') as inp:
                dataDict = pickle.load(inp)
            print(saveDict_name)
            print('dictionary Load successfully to file')


            MP01_list=[MP01_0,MP01_1,MP01_2]
            dictData={}
            dictData_raw={}
            #moco
            dictData['CIRC_ID']=CIRC_ID
            dictData['ID']='moco'
            dictData_raw['CIRC_ID']=CIRC_ID
            dictData_raw['ID']='raw'
            for ss,mp01 in enumerate(MP01_list):
                #Moco

                ind_label=[]
                indtmp=np.shape(mp01._data)[-1]    
                ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
                mask_endo=dataDict[f'Slice{ss}_endo']
                dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
                dictData.update(dicttmp)
                mask_epi=dataDict[f'Slice{ss}_epi']
                dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
                dictData.update(dicttmp)
                #Raw
                ind_label=[]
                indtmp=np.shape(mp01._raw_data)[-1]    
                ind_label=[0,indtmp,indtmp+MP02.Nd,indtmp+MP02.Nd+MP03.Nd]
                mask_endo=dataDict[f'Slice{ss}_endo_raw']
                dicttmp=calc_misArea(mask_endo,ind_label,ss,'endo')
                dictData_raw.update(dicttmp)
                mask_epi=dataDict[f'Slice{ss}_epi_raw']
                dicttmp=calc_misArea(mask_epi,ind_label,ss,'epi')
                dictData_raw.update(dicttmp)
            #########Save the dict
            moco_stats = pd.DataFrame(dictData, index=[0])
            moco_raw_stats = pd.DataFrame(dictData_raw, index=[0])
            filename=os.path.join(defaultPath,f'MisRegristeredArea.csv')
            
            exportData(moco_stats,filename)
            exportData(moco_raw_stats,filename)
    except:
        pass


# %%
