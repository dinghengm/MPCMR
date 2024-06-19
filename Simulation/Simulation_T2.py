#%%
#Function draft on simulate T2 mapping
import random
from Matrix_Basis import *
import numpy as np
import matplotlib.pyplot as plt
from libMapping_v12 import *
from SimulationFunction import *
from scipy.optimize import curve_fit
#%%
#Spin Echo signal with respective to time
def simulateT2(dT=1,T=1000,df=0,T1=1400,Tr=12000,T2=40,TE=40):
    N=int(T/dT)+1
    M=np.zeros((3,N))
    #Use the freePrecess function in the library
    A,B = freePrecess(dT,T1,T2,df)
    #Get the number of point before and after TE
    N1=round(TE/2/dT)
    N2=round((T-TE/2)/dT)

    ### Get Propagation matrix
    Rflip=yrot(np.pi/2)
    #Starting 
    M[:,0]=np.array(np.transpose([0,0,1]))
    Rrefoc = xrot(np.pi)
    M[:,1]=A@Rflip@M[:,0]+B
    #Simulate time >=1
    
    for k in range(2,N1+1,1):
        M[:,k]=np.dot(A,M[:,k-1])+B
    #Refocus
    M[:,N1+1]=A@Rrefoc@M[:,N1]+B
    for k in range(N2-1):
        M[:,k+N1+2]= A@M[:,k+N1+1]+B
    time=np.arange(0,N,1)*dT
    return time,M
# %%
def T2_recovery(tVec,T2,M0):
    #Spin Echo recovery, assuming infinite recovery time
    #Equation: fT2 = @(a)(a(1)*exp(-xData/a(2)) - yDat);
    tVec=np.array(tVec)
    return M0*np.exp(-tVec/T2)
#%%
dT=1       #1 ms
T=1000   #Total Duration 1s
df=0        #10Hz in second
T1=1400
T2=40
TE=40
Tr=12000
SNR=40
#Signal intensity
time,M=simulateT2(dT=dT,T=T,T1=T1,Tr=Tr,T2=T2,TE=TE)
Msig=M[0,:]+ 1j * M[1,:]
plt.plot(time,abs(Msig),'b-')
plt.xlabel('Time (ms)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
#%%
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
        if np.shape(signal)[0]==3:
            sig_avg_val = np.mean(signal[-1,:])
        else:
            sig_avg_val = np.mean(signal)
        sig_avg_db = 10 * np.log10(sig_avg_val)
        # Calculate noise according to [2] then convert to sigma2
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_simga2 = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_signal = np.random.normal(mean_noise, np.sqrt(noise_avg_simga2), N)
        if np.shape(signal)[0]==3:
            tmp[-1,:]=tmp[-1,:]+noise_signal
        else:
            tmp = tmp+noise_signal
    return tmp
# %%
#Fitting:
SNR=40
M_noisy=add_noise(abs(Msig),SNR=SNR,type='Gaussian')
plt.plot(time,M_noisy,'g-.')
plt.legend(['Mz_Noisy'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.grid('True')
plt.show()
plt.close()
# %%
TElist=[30,40,50,60,80,100]
TE_read=M_noisy[TElist]

def sub_mono_t2_fit_exp(ydata,xdata):
    xdata=np.array(xdata)
    ydata=np.array(ydata)
    ydata=abs(ydata)
    ydata=ydata/np.max(ydata)

    t2Init_dif = xdata[0] - xdata[-1]
    try:
        t2Init = t2Init_dif/np.log(ydata[-1]/ydata[0])
        if t2Init<=0:
            t2Init=30
    except:
        t2Init=30
    pdInit=np.max(ydata)*1.5
    params_OLS,params_covariance= curve_fit(T2_recovery,xdata,ydata,method='lm',p0=[t2Init,pdInit],maxfev=5000)
    T2_exp,Mz_exp=params_OLS
    ydata_exp=T2_recovery(xdata,T2_exp,Mz_exp)
    n=len(xdata)
    res=1. / np.sqrt(n) * np.sqrt(np.power(1 - ydata_exp / xdata.T, 2).sum())
    return T2_exp,Mz_exp,res,ydata_exp

# %%

T2_exp,Mz_exp,res,ydata_exp=sub_mono_t2_fit_exp(TE_read,TElist)
print(res)

x_plot=np.arange(start=1,stop=time[-1],step=1)
ydata_exp=abs(T2_recovery(x_plot,T2_exp,Mz_exp))
plt.plot(x_plot,ydata_exp)

#plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
plt.scatter(np.array(TElist),TE_read)
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.title(f'Simulation T2={T2} SNR={SNR}')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.text(x=800,y=0.5,s=f'T2={int(T2_exp)}\ny={Mz_exp:.02f}*e^-t/{int(T2_exp)}')
plt.grid('True')
plt.show()
plt.close()

# %%
T2_simulate=[20,25,30,35,40,45,50,55,80,100]
SNR_list=[20,40,60]
TElist=[30,40,50,60,80,100]
savedir=r'C:\Research\MRI\MP_EPI\Simulation\DifferentT2'
for SNR in SNR_list:
    errors=[]
    T2final_list=[]
    for T2 in T2_simulate:
        time,M=simulateT2(dT=dT,T=T,T1=T1,Tr=Tr,T2=T2,TE=TE)
        Msig=M[0,:]+ 1j * M[1,:]
        M_noisy=add_noise(abs(Msig),SNR=SNR,type='Gaussian')
        TE_read=M_noisy[TElist]
        T2_exp,T2_exp,Mz_exp,res,ydata_exp=sub_mono_t2_fit_exp(TE_read,TElist)

                
        x_plot=np.arange(start=1,stop=time[-1],step=1)
        ydata_exp=abs(T2_recovery(x_plot,T2_exp,Mz_exp))
        plt.plot(x_plot,ydata_exp)

        #plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
        plt.scatter(np.array(TElist),TE_read)
        plt.legend(['Mz_Read'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Magnetization')
        plt.title(f'Simulation T2={T2} SNR={SNR}')
        plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
        plt.text(x=800,y=0.5,s=f'T2={int(T2_exp)}\ny={Mz_exp:.02f}*e^-t/{int(T2_exp)}')
        plt.grid('True')
        plt.show()
        T2final_list.append(T2_exp)
        print(res)
        
    plt.figure()
    bland_altman_plot(T2_simulate,T2final_list,Print_title=f'Simulation Error SNR={SNR}\n')
    plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR}'))
    plt.close()


# %%
