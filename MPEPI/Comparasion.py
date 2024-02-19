#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from scipy.optimize import curve_fit
#sys.path.append('CIRC_AutoCMR_dev/Dingheng/functions')
#from data_fitting import *
import scipy
import scipy.io 

# %%
#Read TIEPI
try:
    nii_img_EPI_TI  = nib.load('EPI_TI_Volume.nii.gz')
    EPI_TI = nii_img_EPI_TI.get_fdata()
    TIList_EPI=[100,180,1105,1185,2135,2355,3135,3955]
    #EPI_T1_map=calculateT1map(EPI_T1,TIList)
    EPI_T1_map=scipy.io.loadmat('/Volumes/Project/Matthew/20230809_1449_CIRC_Phantom_Aug_9th_Diff_/VOLUMES/EPI_T1.mat')
    #READ TISE
    nii_img_SE_T1  = nib.load('SE_TI_Volume.nii.gz')
    SE_T1 = nii_img_SE_T1.get_fdata()
    TIList_SE=[100,180,1105,1185,2105,3105,4105]
    #EPI_T1_map=calculateT1map(EPI_T1,TIList)
    SE_T1_map=np.load('T1_SE.npy')
    #READ TEEPI
    nii_img_EPI_TE  = nib.load('EPI_TE_Volume.nii.gz')
    EPI_TE = nii_img_EPI_TE.get_fdata()
    TEList_EPI=[30,50,80,100,150,180]
    #EPI_T1_map=calculateT1map(EPI_T1,TIList)
    EPI_T2_map=scipy.io.loadmat('/Volumes/Project/Matthew/20230809_1449_CIRC_Phantom_Aug_9th_Diff_/VOLUMES/EPI_T2.mat')
    #READ TESE
    nii_img_SE_TE  = nib.load('SE_TE_Volume.nii.gz')
    SE_TE = nii_img_SE_TE.get_fdata()
    TEList_SE=[30,50,80,100,150]
    #EPI_T1_map=calculateT1map(EPI_T1,TIList)
    SE_T2_map=np.load('T2_SE.npy')
except:
    print('')
#%%
import ipympl
#T2222
fig,ax,cscat= plt_stationary(EPI_T2_map[:,:,2],EPI_TE[:,:,2,:],TEList_EPI,x_read=65,y_read=12,map_type='T2')
#tpc=plt.tripcolor
#plt.colorbar()
updater=plt_update_onclick(fig,ax,EPI_TE[:,:,2,:],TEList_EPI,cscat,map_type='T2')
#%%
#T1111
fig,ax,cscat= plt_stationary(EPI_T1_map[:,:,2],EPI_TI[:,:,2,:],TIList_EPI,x_read=65,y_read=12,map_type='T1')
#tpc=plt.tripcolor
#plt.colorbar()
updater=plt_update_onclick(fig,ax,EPI_TI[:,:,2,:],TIList_EPI,cscat,map_type='T1')
# %%
#information of the plot:
plt.close()
# T1_EPI_ave_unsort=np.array([464,1212,435,1660,1554,583,251,1007,300])
# T1_EPI_std_unsort=np.array([3.9,11,9.64,37,36,7,5,12,10])
T1_EPI_ave_unsort=np.array([459.45,1208.35,434.66,1628.16,1515.93,586.50,237.83,1001.67,290.43])
T1_EPI_std_unsort=np.array([3.76,12.66,3.33,18.39,19.29,5.49,2.77,7.54,35.06])

T1_Molli_ave_unsort=np.array([469,1106,426,1540,1354,565,257,955,297])
# T1_Molli_ave=[426,1106,469,565,1354,1540,297,955,257]
T1_Molli_std_unsort=np.array([7,6,7,8,7,6,4,10,5])

T1_SE_ave_unsort=np.array([506.44,1254.85,444.61,1738,1582,598,181,1032,142])
T1_SE_std_unsort=np.array([9,20,104,58,36,10,7,10,7])
#Reorder it 
arr1inds=T1_EPI_ave_unsort.argsort()
T1_EPI_ave=T1_EPI_ave_unsort[arr1inds]
T1_EPI_std=T1_EPI_std_unsort[arr1inds]
arr2inds=T1_Molli_ave_unsort.argsort()
T1_Molli_ave=T1_Molli_ave_unsort[arr2inds]
T1_Molli_std=T1_Molli_std_unsort[arr2inds]
arr2inds=T1_SE_ave_unsort.argsort()
T1_SE_ave=T1_SE_ave_unsort[arr2inds]
T1_SE_std=T1_SE_std_unsort[arr2inds]

x=np.arange(len(T1_EPI_ave))
plt.errorbar(x,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
plt.errorbar(x,T1_SE_ave,T1_SE_std,label='SE')
plt.errorbar(x,T1_Molli_ave,T1_Molli_std,label='Molli')
plt.xlabel('Tubes Number')
plt.legend()
plt.title('T1 Correlation Plot')
#%%
plt.close()
T2_EPI_ave_unsort=np.array([185,40,37,208,42,36,138,36,36])
T2_EPI_std_unsort=np.array([2.7,1.4,1.1,3.3,0.5,0.7,0.9,0.6,0.8])
# T2_SE_ave=[26,27,33,33,46,41]
# T2_SE_std=[0.6,0.6,0.3,0.3,0.3,0.2]
T2_Flash_ave_unsort=np.array([146,48,43,197,53,42,121,45,44])
T2_Flash_std_unsort=np.array([3.2,0.9,0.6,7.8,0.8,0.6,3.5,0.6,0.4])
T2_SE_ave_unsort=np.array([153.01,42.80,40.88,173.27,45.46,39.48,126.19,40.78,40.07])
T2_SE_std_unsort=np.array([4.01,1.38,1.59,9.3,1.75,2.41,2.86,1.62,1.85])

arr1inds=T2_EPI_ave_unsort.argsort()
T2_EPI_ave=T2_EPI_ave_unsort[arr1inds]
T2_EPI_std=T2_EPI_std_unsort[arr1inds]
arr2inds=T2_Flash_ave_unsort.argsort()
T2_Flash_ave=T2_Flash_ave_unsort[arr2inds]
T2_Flash_std=T2_Flash_std_unsort[arr2inds]
arr2inds=T2_SE_ave_unsort.argsort()
T2_SE_ave=T2_SE_ave_unsort[arr2inds]
T2_SE_std=T2_SE_std_unsort[arr2inds]

x=np.arange(len(T2_EPI_ave))
plt.errorbar(x,T2_EPI_ave,T2_EPI_std,marker='*',label='EPI')
plt.errorbar(x,T2_SE_ave,T2_SE_std,label='SE')
plt.errorbar(x,T2_Flash_ave,T2_Flash_std,label='Flash')

plt.xlabel('Tubes Number')
plt.legend()
plt.title('T2 Correlation Plot')
#%%
#%matplotlib widget
T2_Flash_source=pydicom.read_file('temp/T2/MR000001.dcm')
#Keep T2 in T2-2, and the maxium values are 10 fold difference, so /10
T1_Molli_source=pydicom.read_file('temp/T1/MR000001.dcm')

# T1_mapping=scipy.io.loadmat('temp/T1_mapping.mat')
# T1_mapping=T1_mapping['T1']
# T2_mapping=scipy.io.loadmat('temp/T2_mapping.mat')
# T2_mapping=T2_mapping['T2']
T1_mapping_SE=scipy.io.loadmat('SE_T1.mat')
T1_mapping_SE=T1_mapping_SE['T1']
T2_mapping_SE=scipy.io.loadmat('SE_T2.mat')
T2_mapping_SE=T2_mapping_SE['T2']
T1_mapping_EPI=scipy.io.loadmat('EPI_T1.mat')
T1_mapping_EPI=T1_mapping_EPI['T1']

T2_mapping_EPI=scipy.io.loadmat('EPI_T2.mat')
T2_mapping_EPI=T2_mapping_EPI['T2']
plt.close()
#%%
%matplotlib widget
from scipy import ndimage
T1_Molli=T1_Molli_source.pixel_array
T2_Flash=T2_Flash_source.pixel_array
plt.subplot(131)
plt.imshow(T1_mapping_EPI[32:90,:,3].T,vmin=0,vmax=3000,cmap='magma')
#plt.jet()
plt.axis('off')
plt.title('SE_EPI')
plt.subplot(132)
plt.imshow(T1_mapping_SE[:,:].T,vmin=0,vmax=3000,cmap='magma')
#plt.jet()
plt.axis('off')
plt.title('SE')
plt.subplot(133)
tr = ndimage.rotate(T1_Molli[:,:].T, -90)
a=plt.imshow(tr,vmin=0,vmax=3000,cmap='magma')
#plt.jet()
plt.title('Molli')
plt.axis('off')
plt.colorbar(a)
# %%
plt.close()
plt.subplot(131)
plt.imshow(T2_mapping_EPI[:,:,3].T,vmin=0,vmax=120,cmap='viridis')
#plt.hot()
plt.axis('off')
plt.title('SE_EPI')
plt.subplot(132)
plt.imshow(T2_mapping_SE[:,:].T,vmin=0,vmax=120,cmap='viridis')
#plt.hot()
plt.axis('off')
plt.title('SE')
plt.subplot(133)
tr = ndimage.rotate(T2_Flash[:,:].T, -90)
a=plt.imshow(tr/10,vmin=0,vmax=120,cmap='viridis')
plt.title('FLASH')
#plt.hot()
plt.colorbar(a)
plt.axis('off')
plt.show()

#%%
#Changge the images size
%matplotlib widget
from skimage.transform import resize as imresize
from scipy import ndimage
T1_Molli=T1_Molli_source.pixel_array
T2_Flash=T2_Flash_source.pixel_array
T1_mapping_EPI_temp=T1_mapping_EPI
#newshape0,newshape1,*rest=2*np.shape(T1_mapping_EPI_temp)[0:2]

#T1_mapping_EPI_show=imresize(T1_mapping_EPI_temp,(64,64)
T1_mapping_SE_show=T1_mapping_SE[7:59,3:58]
T1_mapping_MOLLI_show=T1_mapping_SE[7:59,3:58]

fig=plt.figure(figsize=(8,10))
axs0=plt.subplot(141)
im1=plt.imshow(T1_mapping_EPI_show.T,vmin=0,vmax=3000,cmap='magma')
newshape=2*np.shape(T1_mapping_EPI)[0:2]

#plt.jet()
plt.axis('off')
plt.title('SE_EPI')
plt.subplot(142)
im1=plt.imshow(T1_mapping_SE_show.T,vmin=0,vmax=3000,cmap='magma')
#plt.jet()
plt.axis('off')
plt.title('SE')
axs3=plt.subplot(143)
tr = ndimage.rotate(T1_mapping_MOLLI_show.T, -90)
a=plt.imshow(tr,vmin=0,vmax=3000,cmap='magma')
#plt.jet()
plt.title('Molli')
plt.axis('off')
axs4=plt.subplot(144)
cbar1=fig.colorbar(im1,ax=axs4, shrink=1, pad=0.018, aspect=11)
# %%
plt.close()
plt.subplot(131)
plt.imshow(T2_mapping_EPI[:,:,3].T,vmin=0,vmax=120,cmap='viridis')
#plt.hot()
plt.axis('off')
plt.title('SE_EPI')
plt.subplot(132)
plt.imshow(T2_mapping_SE[:,:].T,vmin=0,vmax=120,cmap='viridis')
#plt.hot()
plt.axis('off')
plt.title('SE')
plt.subplot(133)
tr = ndimage.rotate(T2_Flash[:,:].T, -90)
a=plt.imshow(tr/10,vmin=0,vmax=120,cmap='viridis')
plt.title('FLASH')
#plt.hot()
plt.colorbar(a)
plt.axis('off')
plt.show()

#%%
#Plot the correlation plot
%matplotlib inline
plt.figure()
from sklearn.linear_model import LinearRegression
X=T1_SE_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T1_EPI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T1_EPI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T1_SE_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(1100,200,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('MP-EPI T1 (ms)')
plt.legend()
#plt.title('T1 Correlation Plot')

#%%
plt.figure()
from sklearn.linear_model import LinearRegression
X=T1_Molli_ave.reshape((-1, 1))
model = LinearRegression()
model=model.fit(X, T1_EPI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T1_EPI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T1_Molli_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(1100,200,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T1_Molli_ave,T1_EPI_ave,T1_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='MOLLI')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('MOLLI T1 (ms)')
plt.ylabel('MP-EPI T1 (ms)')
plt.legend()


#%%
plt.figure()
from sklearn.linear_model import LinearRegression
X=T1_SE_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T1_Molli_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T1_Molli_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T1_SE_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MOLLI')
plt.text(1100,200,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T1_SE_ave,T1_Molli_ave,T1_Molli_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('Reference T1 (ms)')
plt.ylabel('MOLLI T1 (ms)')
plt.legend()


# %%

def bland_altman_plot(data1, data2, Print_title,*args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    CI_low    = md - 1.96*sd
    CI_high   = md + 1.96*sd

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='black', linestyle='-')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.title(f"{Print_title}")
    plt.xlabel("Means")
    plt.ylabel("Difference")
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


md, sd, mean, CI_low, CI_high = bland_altman_plot(T1_SE_ave, T1_Molli_ave,Print_title='Reference T1, MOLLI T1')

plt.show()
#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(T1_Molli_ave, T1_EPI_ave,Print_title='MOLLI T1, MP-EPI T1')
plt.show()
#%%
md, sd, mean, CI_low, CI_high = bland_altman_plot(T1_SE_ave, T1_EPI_ave,Print_title='Reference T1, MP-EPI T1')

plt.show()

#%%
plt.figure()
from sklearn.linear_model import LinearRegression
X=T2_SE_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T2_EPI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T2_EPI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T2_SE_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(115,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T2_SE_ave,T2_EPI_ave,T2_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('MP-EPI T2 (ms)')
plt.legend()
#%%
plt.figure()
from sklearn.linear_model import LinearRegression
X=T2_Flash_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T2_EPI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T2_EPI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T2_Flash_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='MP-EPI')
plt.text(115,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T2_Flash_ave,T2_EPI_ave,T2_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='FLASH T2')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('FLASH T2 (ms)')
plt.ylabel('MP-EPI T2 (ms)')
plt.legend()
#%%
#Plot the correlation plot --T2
plt.figure()
from sklearn.linear_model import LinearRegression
X=T2_SE_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, T2_Flash_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, T2_Flash_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0,T2_SE_ave[-1],3)
y=a*x + b
plt.plot(x,y,linestyle='dashed',label='FLASH')
plt.text(115,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}')
plt.errorbar(T2_SE_ave,T2_Flash_ave,T2_Flash_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
plt.plot(x,x,linestyle='solid',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlabel('Reference T2 (ms)')
plt.ylabel('FLASH T2 (ms)')
plt.legend()
# %%

md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_Flash_ave, T2_EPI_ave,Print_title='FLASH T2, MP-EPI T2')
plt.show()
md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_SE_ave, T2_EPI_ave,Print_title='Reference T2, MP-EPI T2')
plt.show()
md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_SE_ave, T2_Flash_ave,Print_title='Reference T2, FLASH T2')
plt.show()
# %%
from scipy.stats import linregress
import numpy as np
import plotly.graph_objects as go
def bland_altman_plot(data1, data2, data1_name='A', data2_name='B', subgroups=None, plotly_template='none', annotation_offset=0.05, plot_trendline=True, n_sd=1.96,*args, **kwargs):
    data1 = np.asarray( data1 )
    data2 = np.asarray( data2 )
    mean = np.mean( [data1, data2], axis=0 )
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean( diff )  # Mean of the difference
    sd = np.std( diff, axis=0 )  # Standard deviation of the difference


    fig = go.Figure()

    if plot_trendline:
        slope, intercept, r_value, p_value, std_err = linregress(mean, diff)
        trendline_x = np.linspace(mean.min(), mean.max(), 10)
        fig.add_trace(go.Scatter(x=trendline_x, y=slope*trendline_x + intercept,
                                 name='Trendline',
                                 mode='lines',
                                 line=dict(
                                        width=4,
                                        dash='dot')))
    if subgroups is None:
        fig.add_trace( go.Scatter( x=mean, y=diff, mode='markers', **kwargs))
    else:
        for group_name in np.unique(subgroups):
            group_mask = np.where(np.array(subgroups) == group_name)
            fig.add_trace( go.Scatter(x=mean[group_mask], y=diff[group_mask], mode='markers', name=str(group_name), **kwargs))



    fig.add_shape(
        # Line Horizontal
        type="line",
        xref="paper",
        x0=0,
        y0=md,
        x1=1,
        y1=md,
        line=dict(
            # color="Black",
            width=6,
            dash="dashdot",
        ),
        name=f'Mean {round( md, 2 )}',
    )
    fig.add_shape(
        # borderless Rectangle
        type="rect",
        xref="paper",
        x0=0,
        y0=md - n_sd * sd,
        x1=1,
        y1=md + n_sd * sd,
        line=dict(
            color="SeaGreen",
            width=2,
        ),
        fillcolor="LightSkyBlue",
        opacity=0.4,
        name=f'±{n_sd} Standard Deviations'
    )

    # Edit the layout
    fig.update_layout( title=f'Bland-Altman Plot for {data1_name} and {data2_name}',
                       xaxis_title=f'Average of {data1_name} and {data2_name}',
                       yaxis_title=f'{data1_name} Minus {data2_name}',
                       template=plotly_template,
                       annotations=[dict(
                                        x=1,
                                        y=md,
                                        xref="paper",
                                        yref="y",
                                        text=f"Mean {round(md,2)}",
                                        showarrow=True,
                                        arrowhead=7,
                                        ax=50,
                                        ay=0
                                    ),
                                   dict(
                                       x=1,
                                       y=n_sd*sd + md + annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"+{n_sd} SD",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=-20
                                   ),
                                   dict(
                                       x=1,
                                       y=md - n_sd *sd + annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"-{n_sd} SD",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=20
                                   ),
                                   dict(
                                       x=1,
                                       y=md + n_sd * sd - annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"{round(md + n_sd*sd, 2)}",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=20
                                   ),
                                   dict(
                                       x=1,
                                       y=md - n_sd * sd - annotation_offset,
                                       xref="paper",
                                       yref="y",
                                       text=f"{round(md - n_sd*sd, 2)}",
                                       showarrow=False,
                                       arrowhead=0,
                                       ax=0,
                                       ay=20
                                   )
                               ])
    return fig

#%%