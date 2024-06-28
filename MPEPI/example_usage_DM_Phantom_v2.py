#%%
#Data June 25 2024
from libMapping_v14 import *
import numpy as np
import matplotlib.pyplot as plt 
import os
import scipy.io as sio
from tqdm.auto import tqdm
from imgbasics import imcrop
from skimage.transform import resize as imresize
import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning)
defaultPath= r'C:\Research\MRI\MP_EPI'
plt.rcParams['savefig.dpi'] = 500
import pydicom
import SimpleITK as sitk
#%%
plot = True
CIRC_ID='CIRC_Phantom_T1T2_June_24'
defaultPath=fr'C:\Research\MRI\MP_EPI\Phantom\{CIRC_ID}'
filename=os.path.join(defaultPath,CIRC_ID)+ '.csv'
#%%


dicomPath=os.path.join(defaultPath,'T1_Grappa_new')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_Grappa_new = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
MP01_Grappa_new.go_crop()
MP01_Grappa_new.go_resize(scale=2)

MP01_Grappa_new.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_Grappa_new.go_ir_fit(invertPoint=None)
#%%
MP01_Grappa_new.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_Grappa_new.save()
# %%
dicomPath=os.path.join(defaultPath,'T1_Grappa')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_Grappa = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Grappa_new.cropzone
MP01_Grappa.cropzone=cropzone
MP01_Grappa.go_crop()
MP01_Grappa.go_resize(scale=2)

MP01_Grappa.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_Grappa.go_ir_fit(invertPoint=None)
#%%
MP01_Grappa.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_Grappa.save()
#%%
dicomPath=os.path.join(defaultPath,'T1_NonZoom_new')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_NonZoom = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
MP01_NonZoom.go_crop()
MP01_NonZoom.go_resize(scale=2)

MP01_NonZoom.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_NonZoom.go_ir_fit(invertPoint=None)
#%%
MP01_NonZoom.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_NonZoom.save()
# %%
dicomPath=os.path.join(defaultPath,'T1_SE')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_SE = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
MP01_SE.go_resize(scale=2)

MP01_SE.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_SE.go_ir_fit(invertPoint=None)
#%%
fig=plt.figure()
im = plt.imshow(MP01_SE._map[:,:].squeeze(),vmin=0,vmax=3000,cmap='magma')
plt.axis('off')
fig.colorbar(im)
plt.savefig(os.path.join(os.path.dirname(dicomPath),f'map_{CIRC_ID}_{ID}'))

#%%
MP01_SE.save()
# %%
dicomPath=os.path.join(defaultPath,'T1_Grappa_new_12Point')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_12Points = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Grappa_new.cropzone
MP01_12Points.cropzone=cropzone
MP01_12Points.go_crop()
MP01_12Points.go_resize(scale=2)

MP01_12Points.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_12Points.go_ir_fit(invertPoint=None)
#%%
MP01_12Points.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_12Points.save()
# %%
dicomPath=os.path.join(defaultPath,'T1_Grappa_new_12Point')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_12Points = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Grappa_new.cropzone
MP01_12Points.cropzone=cropzone
MP01_12Points.go_crop()
MP01_12Points.go_resize(scale=2)

MP01_12Points.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_12Points.go_ir_fit(invertPoint=None)
#%%
MP01_12Points.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_12Points.save()
#%%
dicomPath=os.path.join(defaultPath,'T1_Grappa_new_10Point')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_10Points = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Grappa_new.cropzone
MP01_10Points.cropzone=cropzone
MP01_10Points.go_crop()
MP01_10Points.go_resize(scale=2)

MP01_10Points.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_10Points.go_ir_fit(invertPoint=None)
#%%
MP01_10Points.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_10Points.save()
# %%
dicomPath=os.path.join(defaultPath,'T1_Grappa_new_9Point')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_9Points = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Grappa_new.cropzone
MP01_9Points.cropzone=cropzone
MP01_9Points.go_crop()
MP01_9Points.go_resize(scale=2)

MP01_9Points.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_9Points.go_ir_fit(invertPoint=None)
#%%
MP01_9Points.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_9Points.save()
# %%
dicomPath=os.path.join(defaultPath,'T1_Grappa_new_8Point')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_"+ID
MP01_8Points = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Grappa_new.cropzone
MP01_8Points.cropzone=cropzone
MP01_8Points.go_crop()
MP01_8Points.go_resize(scale=2)

MP01_8Points.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_8Points.go_ir_fit(invertPoint=None)
#%%
MP01_8Points.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_8Points.save()
# %%
#Begin to Draw ROI
#This can apply to all 
%matplotlib qt
from roipoly import RoiPoly, MultiRoi
cmap='magma'
crange=[0,3000]
###PLEASE USE SLICE1
image = MP01_12Points._map[:,:,1]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(1))
fig.canvas.manager.set_window_title('Slice '+ str(1))
multirois = MultiRoi(fig=fig,roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())


#%%
Nd=9
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')

# %%
#Add another 2 to the gloabal
roi_names=['0','1']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(1))
fig.canvas.manager.set_window_title('Slice '+ str(1))
multirois = MultiRoi(fig=fig,roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())

for ind,label in enumerate(roi_names):
    add=ind+7
    roi_mask[...,add]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,add]]): .2f} +/- {np.std(image[roi_mask[...,add]]): .2f} um^2/ms')


# %%
#Generate the map values
#save it into an excel spreadsheet
#For the MultiPoints:

import pandas as pd
def export_roi_data(filename,obj,roi_mask,z=None):
    if z==None:
        z=0
    ID=obj.ID
    CIRC_ID=obj.ID
    keys=['CIRC_ID','ID']
    stats=[CIRC_ID,ID]
    #Generate a slice_stats
    slice_stats=[]
    slice_keys=[]
    for i in range(Nd):
        slice_stats.append(np.mean(obj._map[:,:,z][roi_mask[...,i]]))
        slice_keys.append(str(f'ROI {i+1}'))
    for i in range(Nd):
        slice_stats.append(np.std(obj._map[:,:,z][roi_mask[...,i]]))
        slice_keys.append(str(f'ROI {i+1} std'))

    stats.extend(slice_stats)
    keys.extend(slice_keys)

    data=dict(zip(keys,stats))
    data_stats=pd.DataFrame(data,index=[0])
    if os.path.isfile(filename):    
        data_stats.to_csv(filename, index=False, header=False, mode='a')
    else:
        data_stats.to_csv(filename, index=False)

for obj in [MP01_8Points,MP01_9Points,MP01_10Points,MP01_12Points]:
    export_roi_data(filename,obj,roi_mask,z=1)
            

# %%
%matplotlib qt
from roipoly import RoiPoly, MultiRoi
cmap='magma'
crange=[0,3000]
###PLEASE USE SLICE1
####Now let try the SE
image = MP01_SE._map[:,:,0]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(1))
fig.canvas.manager.set_window_title('Slice '+ str(1))
multirois = MultiRoi(fig=fig,roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())


#%%
Nd=9
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')

# %%
#Add another 2 to the gloabal
roi_names=['0','1']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(1))
fig.canvas.manager.set_window_title('Slice '+ str(1))
multirois = MultiRoi(fig=fig,roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())

for ind,label in enumerate(roi_names):
    add=ind+7
    roi_mask[...,add]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,add]]): .2f} +/- {np.std(image[roi_mask[...,add]]): .2f} um^2/ms')


# %%
export_roi_data(filename,MP01_SE,roi_mask)
# %%
T1_bssfp,_,_  = readFolder(os.path.join(defaultPath,r'MR t1map_long_t1_Slice2_MOCO_T1'))
T2_flash,_,_ = readFolder(os.path.join(defaultPath,r'MR t2map_flash_End_MOCO_T2'))

#%%
from imgbasics import imcrop
from skimage.transform import resize as imresize
%matplotlib qt

ID='MP01_Molli_Siemens'
data=T1_bssfp.squeeze()
MP01_Molli_Siemens=mapping(data=np.expand_dims(data,axis=-1),ID=ID,CIRC_ID=CIRC_ID)
MP01_Molli_Siemens.go_crop()


#%%
from roipoly import RoiPoly, MultiRoi
cmap='magma'
crange=[0,3000]
###PLEASE USE SLICE1
####Now let try the SE
MP01_Molli_Siemens._map=MP01_Molli_Siemens._data.squeeze()
image = MP01_Molli_Siemens._map[:,:,0]
roi_names=['0','1','2','3','4','5','6']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(1))
fig.canvas.manager.set_window_title('Slice '+ str(1))
multirois = MultiRoi(fig=fig,roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())


#%%
Nd=9
roi_mask=np.zeros((Nx,Ny,Nd), dtype=bool)
for ind,label in enumerate(roi_names):
    roi_mask[...,ind]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,ind]]): .2f} +/- {np.std(image[roi_mask[...,ind]]): .2f} um^2/ms')

# %%
#Add another 2 to the gloabal
roi_names=['0','1']
fig = plt.figure()
plt.imshow(image, cmap=cmap,vmin=crange[0],vmax=crange[1])
plt.title('Slice '+ str(1))
fig.canvas.manager.set_window_title('Slice '+ str(1))
multirois = MultiRoi(fig=fig,roi_names=roi_names)
Nx,Ny=np.shape(image.squeeze())

for ind,label in enumerate(roi_names):
    add=ind+7
    roi_mask[...,add]=multirois.rois[label].get_mask(image)
    print(fr'Global {int(label)}: {np.mean(image[roi_mask[...,add]]): .2f} +/- {np.std(image[roi_mask[...,add]]): .2f} um^2/ms')


# %%
export_roi_data(filename,MP01_Molli_Siemens,roi_mask,z=1)
#%%
MP01_Molli_Siemens.path=os.path.join(defaultPath,r'MR t1map_long_t1_Slice2_MOCO_T1')
MP01_Molli_Siemens.imshow_map(plot=plot)
MP01_Molli_Siemens.save()

#%%

# %%
#Read the T1mapping on it own, and then fit it
dicomPath=os.path.join(defaultPath,'MR t1map_long_t1_Slice2_MOCO-2')
ID=os.path.abspath(dicomPath).split('\\')[-1]
ID="MP01_Molli_Grid"
MP01_Molli_Grid = mapping(data=dicomPath,reject=False,CIRC_ID=CIRC_ID,ID=ID,searchForMappingObj=True)
#%%
%matplotlib qt
cropzone=MP01_Molli_Siemens.cropzone
MP01_Molli_Grid.cropzone=cropzone
MP01_Molli_Grid.go_crop()

MP01_Molli_Grid.imshow_corrected(plot=plot)
# %%
%matplotlib inline
MP01_Molli_Grid.go_ir_fit(invertPoint=None)
#%%
MP01_Molli_Grid.imshow_map(plot=plot,crange=[0,3000],cmap='magma')
MP01_Molli_Grid.save()



# %%
export_roi_data(filename,MP01_Molli_Grid,roi_mask,z=1)

# %%
#For the difference
#
df=pd.read_csv(filename)
# %%
'''
In the dataframe, everything is under certain order

'''
#Mean Value
Nd=9
Map_8_point=[]
Map_9_point=[]
Map_10_point=[]
Map_12_point=[]
Map_SE=[]
Map_Molli_Siemens=[]
Map_Molli_Grid=[]
#Std value
Map_8_point_std=[]
Map_9_point_std=[]
Map_10_point_std=[]
Map_12_point_std=[]
Map_SE_std=[]
Map_Molli_Siemens_std=[]
Map_Molli_Grid_std=[]

mapValueList=[Map_8_point,Map_9_point,Map_10_point,Map_12_point,Map_SE,Map_Molli_Siemens,Map_Molli_Grid]
mapValueStdList=[Map_8_point_std,Map_9_point_std,Map_10_point_std,Map_12_point_std,Map_SE_std,Map_Molli_Siemens_std,Map_Molli_Grid_std]
for nn,maptmp in enumerate(mapValueList):

    for i in range(Nd):
        df_tmp=df.iloc[nn]

        maptmp.append(df_tmp[f'ROI {int(i+1)}'])
        mapValueStdList[nn].append(df_tmp[f'ROI {int(i+1)} std'])
#%%
def bland_altman_rel_plot(data1, data2, Print_title,*args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = (data1 - data2)/data2*100                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    #plt.title(r"$\mathbf{Bland-Altman}$" + " " + r"$\mathbf{Plot}$"+f"\n {Print_title}")
    #plt.xlabel("Means")
    #plt.ylabel("Difference")
    plt.ylim(md - 3.5*sd, md + 3.5*sd)

    xOutPlot = np.min(mean) + (np.max(mean)-np.min(mean))*1.14

    plt.text(xOutPlot, md - 1.96*sd, 
        r'-1.96SD:' + "\n" + "%.2f" % CI_low, 
        ha = "center",
        va = "center",
        )
    plt.text(xOutPlot, md + 1.96*sd, 
        r'+1.96SD:' + "\n" + "%.2f" % CI_high, 
        ha = "center",
        va = "center",
        )
    plt.text(xOutPlot, md, 
        r'Mean:' + "\n" + "%.2f" % md, 
        ha = "center",
        va = "center",
        )
    plt.subplots_adjust(right=0.85)

    return md, sd, mean, CI_low, CI_high
#%%
#Always use data 2 as the standard (x axis)
def plotRegression(data1,data2):
    from sklearn.linear_model import LinearRegression
    Y=np.array(data1).reshape((-1,1))
    X=np.array(data2).reshape((-1,1))

    model = LinearRegression()
    model=model.fit(X, Y)
    #plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
    r_sq = model.score(X, Y)
    a=model.coef_[0]
    b=model.intercept_
    x=np.arange(min(X),max(X),3)
    y=a*x + b
    #print(a,b,r_sq)

    plt.plot(x,y,linestyle='solid')
    plt.scatter(data2,data1)
    #plt.errorbar(all_data_T1[0],all_data_T1[1],np.std(all_data_T1[1]), ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
    #x=np.arange(0,max(X)+50,3)
    #plt.plot(x,x,linestyle='dashed',label='Reference')
    plt.xlim=((-5,x[-1]+50))
    plt.legend()
    return a,b,r_sq
#%%
i=8
plt.style.use('seaborn')
plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_rel_plot(Map_8_point, Map_SE,Print_title='MP_EPI-MOLLI')
plt.xlabel("Mean (8 point MP-EPI's T1, SE T1) [ms]",fontsize=14)
plt.ylabel("Reletive Error %",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Bland'))
plt.show()
i=9
plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_rel_plot(Map_9_point, Map_SE,Print_title='MP_EPI-MOLLI')
plt.xlabel("Mean (9 point MP-EPI's T1, SE T1) [ms]",fontsize=14)
plt.ylabel("Reletive Error %",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Bland'))
i=10
plt.show()
plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_rel_plot(Map_10_point, Map_SE,Print_title='MP_EPI-MOLLI')
plt.xlabel("Mean (10 point MP-EPI's T1, SE T1) [ms]",fontsize=14)
plt.ylabel("Reletive Error %",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Bland'))
i=12
plt.show()
plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_rel_plot(Map_12_point, Map_SE,Print_title='MP_EPI-MOLLI')
plt.xlabel("Mean (12 point MP-EPI's T1, SE T1) [ms]",fontsize=14)
plt.ylabel("Reletive Error %",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Bland'))
plt.show()
#%%
plt.close()
i=8
a,b,r_sq=plotRegression(Map_8_point,Map_SE)
plt.text(600,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.xlabel('SE T1 (ms)',fontsize=14)
plt.ylabel(f"{i} point MP-EPI's T1 (ms)",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Regression'))
plt.show()
i=9
a,b,r_sq=plotRegression(Map_9_point,Map_SE)
plt.text(600,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.xlabel('SE T1 (ms)',fontsize=14)
plt.ylabel(f"{i} point MP-EPI's T1 (ms)",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Regression'))
plt.show()
i=10
a,b,r_sq=plotRegression(Map_10_point,Map_SE)
plt.text(600,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.xlabel('SE T1 (ms)',fontsize=14)
plt.ylabel(f"{i} point MP-EPI's T1 (ms)",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Regression'))
plt.show()
i=12
a,b,r_sq=plotRegression(Map_12_point,Map_SE)
plt.text(600,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.xlabel('SE T1 (ms)',fontsize=14)
plt.ylabel(f"{i} point MP-EPI's T1 (ms)",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'{i}Point_Regression'))
plt.show()


#%%
#BlandAltman

a,b,r_sq=plotRegression(Map_Molli_Siemens,Map_SE)
plt.text(600,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.xlabel('SE T1 (ms)',fontsize=18)
plt.ylabel(f"Molli_Siemens T1 (ms)",fontsize=18)
plt.savefig(os.path.join(defaultPath,'Statitics',f'Molli_Siemens_Regression'))
plt.show()
a,b,r_sq=plotRegression(Map_Molli_Grid,Map_SE)
plt.text(600,1300,f'y={round(a[0],3)}x+{round(b[0],2)}\nR={round(r_sq,3)}')
plt.xlabel('SE T1 (ms)',fontsize=14)
plt.ylabel(f"Molli_Grid T1 (ms)",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'Molli_Grid_Regression'))
plt.show()

plt.show()
plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_rel_plot(Map_Molli_Siemens, Map_SE,Print_title='MP_EPI-MOLLI')
plt.xlabel("Mean (Molli_Siemens T1, SE T1) [ms]",fontsize=14)
plt.ylabel("Reletive Error %",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'Molli_Siemens_Bland'))
plt.show()
plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_rel_plot(Map_Molli_Grid, Map_SE,Print_title='MP_EPI-MOLLI')
plt.xlabel("Mean (Molli_Grid T1, SE T1) [ms]",fontsize=14)
plt.ylabel("Reletive Error %",fontsize=14)
plt.savefig(os.path.join(defaultPath,'Statitics',f'Molli_Grid_Bland'))
plt.show()



# %%
Map_8_point_Abs=[(Map_8_point[i]-Map_SE[i]) for i in range(len(Map_SE))]
Map_9_point_Abs=[(Map_9_point[i]-Map_SE[i]) for i in range(len(Map_SE))]
Map_10_point_Abs=[(Map_10_point[i]-Map_SE[i]) for i in range(len(Map_SE))]
Map_12_point_Abs=[(Map_12_point[i]-Map_SE[i]) for i in range(len(Map_SE))]
Map_SE_Abs=[(Map_SE[i]-Map_SE[i]) for i in range(len(Map_SE))]
Map_Molli_Siemens_Abs=[(Map_Molli_Siemens[i]-Map_SE[i]) for i in range(len(Map_SE))]
Map_Molli_Grid_Abs=[(Map_Molli_Grid[i]-Map_SE[i]) for i in range(len(Map_SE))]


Map_8_point_Rel=[(Map_8_point[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
Map_9_point_Rel=[(Map_9_point[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
Map_10_point_Rel=[(Map_10_point[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
Map_12_point_Rel=[(Map_12_point[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
Map_SE_Rel=[(Map_SE[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
Map_Molli_Siemens_Rel=[(Map_Molli_Siemens[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
Map_Molli_Grid_Rel=[(Map_Molli_Grid[i]-Map_SE[i])/Map_SE[i]*100 for i in range(len(Map_SE))]
%matplotlib inline

plt.style.use('seaborn')
plt.figure()
xaxis=Map_SE
plt.scatter(xaxis,Map_8_point_Rel,c='g',label='8Points')
plt.scatter(xaxis,Map_9_point_Rel,c='b',label='9Points')
plt.scatter(xaxis,Map_10_point_Rel,c='y',label='10Points')
plt.scatter(xaxis,Map_12_point_Rel,c='k',label='12Points')
plt.scatter(xaxis,Map_SE_Rel,c='r',label='Reference')
plt.legend(loc='lower left')
plt.ylim(-5,5)
plt.xlabel('T1 [ms]')
plt.ylabel('Reletive Error %')
plt.show()
plt.figure()
xaxis=Map_SE
plt.scatter(xaxis,Map_SE_Rel,c='r',label='Reference')
plt.scatter(xaxis,Map_Molli_Siemens_Rel,c='b',marker='s',label='Molli')
plt.scatter(xaxis,Map_Molli_Grid_Rel,c='y',marker='o',label='Molli_2')
plt.legend(loc='lower left')
plt.xlabel('T1 [ms]')
plt.ylabel('Reletive Error %')
plt.ylim(-20,5)
plt.show()
# %%
