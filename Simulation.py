#%%
import random
from Matrix_Basis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from libMapping_v12 import *
from SimulationFunction import *
#Make the simulation function:
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
    #PAPER TO READ
    if type=='Gaussian':
        # Set a target SNR
        target_snr_db = SNR
        # Calculate signal power and convert to dB 
        sig_avg_val = np.mean(signal[-1,:])
        sig_avg_db = 10 * np.log10(sig_avg_val)
        # Calculate noise according to [2] then convert to sigma2
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_simga2 = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_signal = np.random.normal(mean_noise, np.sqrt(noise_avg_simga2), N)
        tmp[-1,:]=tmp[-1,:]+noise_signal
    return tmp
def ir_recovery(tVec,T1,ra,rb):
    #Equation: abs(ra+rb *exp(-tVec(TI)/T1))
    tVec=np.array(tVec)
    #Return T1Vec,ra,rb
    return ra + rb* np.exp(-tVec/T1)

# %%
dT=1       #1 ms
T=10000   #Total Duration 1s
df=0        #10Hz in second
T1=1400
T2=40
Te=40
Tr=12000
SNR=20
flip_angle=np.pi
time,M=simulateT1(dT=dT,T=T,df=df,T1=T1,Tr=Tr,flip_angle=flip_angle)
plt.plot(time,np.abs(M[2,:]),'b-')
plt.show()
#plt.plot(time,np.abs(M[1,:]),'r--')
M_noisy=add_noise(M,SNR=SNR,type='Gaussian')
plt.plot(time,np.abs(M_noisy[2,:]),'g-.')
plt.legend(['Mz_Noisy'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.grid('True')
plt.show()
plt.close()

TIlist=[110,190,240,320,340,440,840,940,1240,1540,2140,2220,2740,2820,3140,4140,5140]
TI_read=M_noisy[:,TIlist]
#How to fix the 

#%%
def sub_ir_fit_grid(data=None,TIlist=None,rabound=[1e-4,1e4],rbbound=[-10000,0],T1bound=[1,5000]):
    ###From rdNlsPr from qmrlab
    if np.size(data) != np.size(TIlist):
        return
    T1Start = T1bound[0]
    T1Stop= T1bound[1]
    #If Zoom it could be different, by default it's 1
    TIarray=np.array(TIlist)
    T1Vec = np.matrix(np.arange(T1Start, T1Stop+1, 1, dtype=np.float))
    Nlen=np.size(TIarray)
    the_exp = np.exp(-TIarray[:,np.newaxis] * 1/T1Vec)
    exp_sum = 1. / Nlen * the_exp.sum(0).T
    rho_norm_vec = np.sum(np.power(the_exp,2), 0).T - 1./Nlen*np.power(the_exp.sum(0).T,2)
    data = np.matrix(data.ravel()).T
    n = data.shape[0]
    y_sum = data.sum()
    rho_ty_vec = (data.T * the_exp).T - exp_sum * y_sum 
    res=np.power(np.abs(rho_ty_vec), 2)/rho_norm_vec
    maxInd=np.argmax(res)

    T1_exp=T1Vec[0,maxInd]
    rb_exp=rho_ty_vec[maxInd,0] / rho_norm_vec[maxInd,0]
    ra_exp=1./Nlen*(y_sum - rb_exp*the_exp[:, maxInd].sum())
    ydata_exp=ir_recovery(TIarray,T1_exp,ra_exp,rb_exp)
    res=chisquareTest(data,ydata_exp)
    return T1_exp,ra_exp,rb_exp,res,ydata_exp


T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(TI_read[-1,:],TIlist)


x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
ydata_exp=abs(ir_recovery(x_plot,T1_exp,ra_exp,rb_exp))
plt.plot(x_plot,ydata_exp)

#plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
plt.scatter(np.array(TIlist),TI_read[2,:])
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.title(f'Simulation T1={T1} SNR={SNR}')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.text(x=8000,y=0,s=f'T1={int(T1_exp)}\ny={ra_exp:.02f}+{rb_exp:.02f}*e^-t/{int(T1_exp)}')
plt.grid('True')
plt.show()

#%%
def sub_ir_fit_grid_WLS(data=None,TIlist=None,rabound=[1e-4,1e4],rbbound=[-10000,0],T1bound=[1,5000],error='l1',Niter=1):
    T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(data=data,TIlist=TIlist,rabound=rabound,rbbound=rbbound,T1bound=T1bound)
    for _ in range(Niter):
        if error=='l1':
            simga_square=abs(ydata_exp-data)
        elif error=='l2':
            simga_square=abs(ydata_exp-data)**2
        weights = 1 / (simga_square)
        #data_tmp=data*weights
        #T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(data=data_tmp,TIlist=TIlist,rabound=rabound,rbbound=rbbound,T1bound=T1bound)
        params_WLS,params_covariance = curve_fit(ir_recovery,TIlist,data,method='lm',p0=[T1_exp,ra_exp,rb_exp],maxfev=5000,sigma=weights,absolute_sigma=True)
        #print(weights)
        T1_exp,ra_exp,rb_exp=params_WLS
    ydata_exp=ir_recovery(TIlist,T1_exp,ra_exp,rb_exp)
    res=chisquareTest(data,ydata_exp)
    return T1_exp,ra_exp,rb_exp,res,ydata_exp


T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid_WLS(TI_read[-1,:],TIlist)


x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
ydata_exp=abs(ir_recovery(x_plot,T1_exp,ra_exp,rb_exp))
plt.plot(x_plot,ydata_exp)

#plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
plt.scatter(np.array(TIlist),TI_read[2,:])
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.title(f'Simulation T1={T1} SNR={SNR}')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.text(x=8000,y=0,s=f'T1={int(T1_exp)}\ny={ra_exp:.02f}+{rb_exp:.02f}*e^-t/{int(T1_exp)}')
plt.grid('True')
plt.show()




#%%
T1_final,ra_final,rb_final,resTmps,returnInd=ir_fit(abs(TI_read[-1,:]),TIlist,type='WLS',Niter=2,searchtype='lm')

x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
ydata_exp=abs(ir_recovery(x_plot,T1_final,ra_final,rb_final))
plt.plot(x_plot,ydata_exp)

plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
plt.scatter(np.array(TIlist),np.abs(TI_read[2,:]))
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.title(f'Simulation T1={T1} SNR={SNR}')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.text(x=8000,y=0,s=f'T1={int(T1_final)}\ny={ra_final:.02f}+{rb_final:.02f}*e^-t/{int(T1_final)}')
plt.grid('True')
plt.show()
# %%
T1_simulate=[1000,1100,1200,1300,1400,1500,1600]
TIlist=[110,190,240,320,340,440,840,940,1240,1540,2140,2220,2740,2820,3140,4140,5140]

SNR_list=[20,30,40,50]
savedir=r'C:\Research\MRI\MP_EPI\Simulation\l1Nrom'

for SNR in SNR_list:
    errors=[]
    T1final_list_WLS=[]
    T1final_list_OLS=[]
    for T1 in T1_simulate:
        time,M=simulateT1(dT=dT,T=T,df=df,T1=T1,T2=T2,Tr=Tr,flip_angle=flip_angle)
        M_noisy=add_noise(abs(M),SNR=SNR,type='Gaussian')
        TI_read=M_noisy[:,TIlist]
        for type in ['OLS','WLS']:
            T1_final,ra_final,rb_final,resTmps,returnInd=ir_fit(TI_read[-1,:],TIlist,type=type,Niter=2,error='l1')
            x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
            ydata_exp=abs(ir_recovery(x_plot,T1_final,ra_final,rb_final))
            plt.figure()
            plt.plot(x_plot,ydata_exp)

            plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
            plt.scatter(np.array(TIlist),np.abs(TI_read[2,:]))
            plt.legend(['Mz_Read'])
            plt.xlabel('Time (ms)')
            plt.ylabel('Magnetization')
            plt.title(f'Simulation T1={T1} SNR={SNR}\n  {type}')
            plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
            plt.text(x=8000,y=0,s=f'{type}\nT1={int(T1_final)}\ny={ra_final:.02f}+{rb_final:.02f}*e^-t/{int(T1_final)}')
            plt.grid('True')
            errors.append(resTmps)
            if type =='OLS':
                T1final_list_OLS.append(T1_final)
            if type =='WLS':
                T1final_list_WLS.append(T1_final)
            #plt.savefig(os.path.join(savedir,f'Simulation T1={T1} SNR={SNR} WLS'))
            plt.savefig(os.path.join(savedir,f'Simulation T1={T1} SNR={SNR} {type}'))
            plt.close()
    error=[min(i) for i in errors]
    plt.figure()
    bland_altman_plot(T1_simulate,T1final_list_OLS,Print_title=f'Simulation Error SNR={SNR}\n  OLS')
    plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} OLS'))
    plt.close()
    plt.figure()
    bland_altman_plot(T1_simulate,T1final_list_WLS,Print_title=f'Simulation Error SNR={SNR}\n  WLS')
    plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} WLS'))
    plt.close()
# %%
