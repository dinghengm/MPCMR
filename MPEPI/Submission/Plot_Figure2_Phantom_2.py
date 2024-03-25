#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pydicom
from scipy.optimize import curve_fit
import scipy
import scipy.io 

# %%
#information of the plot:
plt.close()

#Ref
DWI_Ref_ave_unsort=np.array([1.11,1.04,1.03,0.99,0.90,0.71,0.69])
DWI_Ref_std_unsort=np.array([0.01,0.02,0.01,0.01,0.03,0.02,0.05])

#1

DWI_ave_unsort=np.array([1.11,1.04,1.03,0.99,0.89,0.71,0.68])
DWI_std_unsort=np.array([0.01,0.02,0.01,0.01,0.03,0.02,0.05])

#DWI_ave_unsort=np.array([1.43,1.36,1.36,1.31,1.21,1.03,0.98])
#DWI_std_unsort=np.array([0.02,0.01,0.01,0.01,0.02,0.02,0.04])


arr1inds=DWI_Ref_ave_unsort.argsort()
DWI_Ref_ave=DWI_Ref_ave_unsort[arr1inds]
DWI_Ref_std=DWI_Ref_std_unsort[arr1inds]

arr1inds=DWI_ave_unsort.argsort()
DWI_ave=DWI_ave_unsort[arr1inds]
DWI_std=DWI_std_unsort[arr1inds]

x=np.arange(len(DWI_Ref_ave_unsort))
plt.errorbar(x,DWI_ave,DWI_std,marker='*',label='ADC')
plt.errorbar(x,DWI_Ref_ave,DWI_Ref_std,label='Identity')
plt.xlabel('Tubes Number')
plt.legend()
plt.title('ADC Correlation Plot')

#%%
#Plot the correlation plot
%matplotlib inline
from sklearn.linear_model import LinearRegression
X=DWI_Ref_ave.reshape((-1, 1))

model = LinearRegression()
model=model.fit(X, DWI_ave)

model = LinearRegression()
model=model.fit(X, DWI_ave)
#plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std,marker='*',label='EPI')
r_sq = model.score(X, DWI_ave)
a=model.coef_[0]
b=model.intercept_
x=np.arange(0.6,DWI_Ref_ave[-1]+0.1,0.01)
y=a*x + b
plt.plot(x,y,linestyle='solid',label='MP-EPI')
plt.text(0.88,0.7,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}',fontsize=18)
plt.errorbar(DWI_Ref_ave,DWI_ave,DWI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
#plt.plot(x,x,linestyle='dashed',label='Identity')
plt.xlim(0.6,DWI_Ref_ave[-1]+0.1)
plt.ylim(0.6,DWI_Ref_ave[-1]+0.1)
plt.xlabel('Reference ADC [µm2/ms]',fontsize=18)
plt.ylabel('MP-EPI ADC [µm2/ms]',fontsize=18)
plt.legend(fontsize=18)
#plt.title('T1 Correlation Plot')




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
    #plt.title(f"{Print_title}",fontsize=14)

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

plt.figure()
md, sd, mean, CI_low, CI_high = bland_altman_plot(DWI_Ref_ave, DWI_ave,Print_title='Reference ADC, MP-EPI ADC')
plt.xlabel("Mean (Reference ADC, MP-EPI ADC) [µm2/ms]",fontsize=18)
plt.ylabel("ΔADC [µm2/ms]",fontsize=18)
plt.show()
#%%
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
x=np.arange(0,1800,3)
y=a*x + b
plt.plot(x,y,linestyle='solid',label='MP-EPI')
plt.text(900,200,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}',fontsize=18)
plt.errorbar(T1_SE_ave,T1_EPI_ave,T1_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
#plt.plot(x,x,linestyle='dashed',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlim(0,1800)
plt.ylim(0,1800)
plt.xlabel('Reference T1 [ms]',fontsize=18)
plt.ylabel('MP-EPI T1 [ms]',fontsize=18)
plt.legend(fontsize=18)
#plt.title('T1 Correlation Plot')


# %%



md, sd, mean, CI_low, CI_high = bland_altman_plot(T1_SE_ave, T1_EPI_ave,Print_title='Reference T1, MOLLI T1')
plt.xlabel("Mean (Reference T1, MP-EPI T1) [ms]",fontsize=18)
plt.ylabel("ΔT1 [ms]",fontsize=18)
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
x=np.arange(0,220,3)
y=a*x + b
plt.plot(x,y,linestyle='solid',label='MP-EPI')
plt.text(100,30,f'y={a:.3f}x+{b:.3f}\nR={r_sq:.3f}',fontsize=18)
plt.errorbar(T2_SE_ave,T2_EPI_ave,T2_EPI_std, ls='none',ecolor='blue',color='blue',fmt='.', markersize='10',capsize=4, elinewidth=2)
#plt.plot(x,x,linestyle='dashed',label='Reference')
#plt.xlim=((-5,x[-1]+50))
plt.xlim(0,220)
plt.ylim(0,220)
plt.tick_params(axis='both', which='minor', labelsize=18)
plt.xlabel('Reference T2 (ms)',fontsize=18)
plt.ylabel('MP-EPI T2 (ms)',fontsize=18)
plt.legend(fontsize=18)

# %%

md, sd, mean, CI_low, CI_high = bland_altman_plot(T2_SE_ave, T2_EPI_ave,Print_title='FLASH T2, MP-EPI T2')
plt.xlabel("Mean (Reference T2, MP-EPI T2) [ms]",fontsize=18)
plt.ylabel("ΔT2 [ms]",fontsize=18)
plt.show()

# %%
