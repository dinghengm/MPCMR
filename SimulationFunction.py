import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy
import random
from Matrix_Basis import *
import os
def get_p_value(data):
    #In the order of 0-1, 1-2, 1-3
    try:
        pvalues=[]
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[1])
        pvalues.append(pvalue)
        stat,pvalue=scipy.stats.ttest_ind(data[1],data[2])
        pvalues.append(pvalue)
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[2])
        pvalues.append(pvalue)
    except:
        pvalues=[]
        stat,pvalue=scipy.stats.ttest_ind(data[0],data[1])
        pvalues.append(pvalue)
    return pvalues
def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


# %% Claim Bland Altman plot
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
    plt.title(r"$\mathbf{Bland-Altman}$" + " " + r"$\mathbf{Plot}$"+f"\n {Print_title}")
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
def simulateT1(dT=1,T=10000,df=0,T1=1400,Tr=12000,T2=40,flip_angle=np.pi):
    N=int(T/dT)+1
    M=np.zeros((3,N))
    A,B = freePrecess(dT,T1,T2,df)
    Ry=yrot(flip_angle)
    M[:,0]=np.array(np.transpose([0,0,1]))
    M[:,0]=np.dot(M[:,0],Ry)
    for k in range(1,N,1):
        M[:,k]=np.dot(A,M[:,k-1])+B
        if (k % Tr==0):
        #M[:,k]=np.dot(A,M[:,k-1])+B
            M[:,k]=np.dot(M[:,k-1],Ry)
    time=np.arange(0,N,1)*dT
    return time,M

#Add noise to the Mz
#What is the noise level
#Add the salt and pepper noise
def add_noise(signal,p=0.05,q=0.05,type='SaltPepper',SNR=20):
    N=np.shape(signal)[-1] 
    tmp=np.copy(signal)
    #If p, then add the max
    max=np.max(tmp)
    #If q, then add the min
    min=np.min(tmp)
    if type=='SaltPepper':
        for i in range(N):
            probability=random.randint(0,100)/100
            if probability<p:
                tmp[-1,i]=max
            elif probability>p and probability<(p+q):
                tmp[-1,i]=min
    if type=='Gaussian':
        # Set a target SNR
        target_snr_db = SNR
        # Calculate signal power and convert to dB 
        sig_avg_val = np.mean(signal[-1,:])
        sig_avg_db = 10 * np.log10(sig_avg_val)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_signal = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), N)
        tmp[-1,:]=tmp[-1,:]+noise_signal
    return tmp
def ir_recovery(tVec,T1,ra,rb):
    #Equation: abs(ra+rb *exp(-tVec(TI)/T1))
    tVec=np.array(tVec)
    #Return T1Vec,ra,rb
    return ra + rb* np.exp(-tVec/T1)
