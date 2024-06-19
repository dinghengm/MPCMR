#%%
import random
from Matrix_Basis import *
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy 
import sys
sys.path.append('./MPEPI')
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


T1_exp,ra_exp,rb_exp,res,ydata_exp=sub_ir_fit_grid(TI_read[-1,:],TIlist)
print(res)

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
plt.close()


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
plt.close()
# %%
T1_simulate=[800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800]
T1_simulate=[1400,1500,1600,1700,1800]

TIlist=[110,190,240,320,340,440,840,940,1240,1540,2140,2220,2740,2820,3140,4140,5140]

SNR_list=[20]
savedir=r'C:\Research\MRI\MP_EPI\Simulation\l1Nrom'

for SNR in SNR_list:
    errors=[]
    T1final_list_WLS=[]
    T1final_list_OLS=[]
    T1final_list_Grid=[]
    for T1 in T1_simulate:
        time,M=simulateT1(dT=dT,T=T,df=df,T1=T1,T2=T2,Tr=Tr,flip_angle=flip_angle)
        M_noisy=add_noise(abs(M),SNR=SNR,type='Gaussian')
        TI_read=M_noisy[:,TIlist]
        for type in ['OLS','WLS']:
            T1_final,ra_final,rb_final,resTmps,returnInd=ir_fit(TI_read[-1,:],TIlist,searchtype='lm',type=type,Niter=2,error='l1',invertPoint=4)
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
            print(resTmps)
            plt.show()
            if type =='OLS':
                T1final_list_OLS.append(T1_final)
            if type =='WLS':
                T1final_list_WLS.append(T1_final)
            #plt.savefig(os.path.join(savedir,f'Simulation T1={T1} SNR={SNR} WLS'))
            #plt.savefig(os.path.join(savedir,f'Simulation T1={T1} SNR={SNR} {type}'))
            #plt.close()
        T1_final,ra_final,rb_final,res,returnInd=ir_fit(abs(TI_read[-1,:]),TIlist,searchtype='grid',T1bound=[1,5000],invertPoint=4)
        x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
        ydata_exp=abs(ir_recovery(x_plot,T1_final,ra_final,rb_final))
        
        plt.figure()
        plt.plot(x_plot,ydata_exp)
        plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
        plt.scatter(np.array(TIlist),np.abs(TI_read[2,:]))
        plt.legend(['Mz_Read'])
        plt.xlabel('Time (ms)')
        plt.ylabel('Magnetization')
        plt.title(f'Simulation T1={T1} SNR={SNR}\n  Grid')
        plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
        plt.text(x=8000,y=0,s=f'{type}\nT1={int(T1_final)}\ny={ra_final:.02f}+{rb_final:.02f}*e^-t/{int(T1_final)}')
        plt.grid('True')
        T1final_list_Grid.append(T1_final)
        print(resTmps)
        plt.show()
    #error=[min(i) for i in errors]
    plt.figure()
    bland_altman_plot(T1_simulate,T1final_list_OLS,Print_title=f'Simulation Error SNR={SNR}\n  OLS')
    #plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} OLS'))
    #plt.close()
    plt.figure()
    bland_altman_plot(T1_simulate,T1final_list_WLS,Print_title=f'Simulation Error SNR={SNR}\n  WLS')
    plt.figure()
    bland_altman_plot(T1_simulate,T1final_list_Grid,Print_title=f'Simulation Error SNR={SNR}\n  Grid')
    #plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} WLS'))
    #plt.close()

#%%
T1=1300
TIlist1=[110,190,240,320,340,840,940,1240,1540,2140,2220,2740]
TIlist8000=[110,190,240,320,340,840,940,1240,1540,2140,2220,2740,8000]
SNR=20
time,M=simulateT1(dT=dT,T=T,df=df,T1=T1,T2=T2,Tr=Tr,flip_angle=flip_angle)
M_noisy=add_noise(abs(M),SNR=SNR,type='Gaussian')
TI_read=M_noisy[:,TIlist1]
TI_read_8000=np.append(TI_read,M_noisy[:,40,np.newaxis],axis=-1)
T1_final_ols,ra_final_ols,rb_final_ols,_,returnInd=ir_fit(abs(TI_read[-1,:]),TIlist1,searchtype='lm',ra=1,rb=-2,T1=1200,Niter=2,type='OLS',error='l1')
T1_final_wls,ra_final_wls,rb_final_wls,_,_=ir_fit(abs(TI_read[-1,:]),TIlist1,searchtype='lm',ra=1,rb=-2,T1=1200,Niter=2,type='WLS',error='l1')
T1_final_grid,ra_final_grid,rb_final_grid,_,_=ir_fit(abs(TI_read[-1,:]),TIlist1,searchtype='grid',T1bound=[1,5000])
x_plot=np.arange(start=1,stop=TIlist[-1],step=1)
ydata_exp_ols=abs(ir_recovery(x_plot,T1_final_ols,ra_final_ols,rb_final_ols))
ydata_exp_wls=abs(ir_recovery(x_plot,T1_final_wls,ra_final_wls,rb_final_wls))
ydata_exp_grid=abs(ir_recovery(x_plot,T1_final_grid,ra_final_grid,rb_final_grid))
plt.scatter(np.array(TIlist1),TI_read[2,:])
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.title(f'Simulation T1={T1} SNR={SNR}')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.text(x=8000,y=0,s=f'OLS:y={ra_final_ols:.02f}+{rb_final_ols:.02f}*e^-t/{int(T1_final_ols)}')
plt.text(x=8000,y=-.25,s=f'WLS:y={ra_final_wls:.02f}+{rb_final_wls:.02f}*e^-t/{int(T1_final_wls)}')
plt.text(x=8000,y=-0.5,s=f'Grid:y={ra_final_grid:.02f}+{rb_final_grid:.02f}*e^-t/{int(T1_final_grid)}')
plt.grid('True')
plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read[2,:][0:returnInd]),color='r')
plt.plot(x_plot,ydata_exp_ols,label='OLS')
plt.plot(x_plot,ydata_exp_wls,label='WLS')
plt.plot(x_plot,ydata_exp_grid,label='Grid')
plt.legend()
plt.show()
#%%
TIlist8000=[110,190,240,320,340,840,940,1240,1540,2140,2220,2740,8000]
TI_read_8000=np.append(TI_read,M_noisy[:,40,np.newaxis],axis=-1)
T1_final_ols,ra_final_ols,rb_final_ols,_,returnInd=ir_fit(abs(TI_read_8000[-1,:]),TIlist8000,searchtype='lm',ra=1,rb=-2,T1=1200,Niter=2,type='OLS',error='l1')
T1_final_wls,ra_final_wls,rb_final_wls,_,_=ir_fit(abs(TI_read_8000[-1,:]),TIlist8000,searchtype='lm',ra=1,rb=-2,T1=1200,Niter=2,type='WLS',error='l1')
T1_final_grid,ra_final_grid,rb_final_grid,_,_=ir_fit(abs(TI_read_8000[-1,:]),TIlist8000,searchtype='grid',T1bound=[1,5000])
x_plot=np.arange(start=1,stop=TIlist8000[-1],step=1)
ydata_exp_ols=abs(ir_recovery(x_plot,T1_final_ols,ra_final_ols,rb_final_ols))
ydata_exp_wls=abs(ir_recovery(x_plot,T1_final_wls,ra_final_wls,rb_final_wls))
ydata_exp_grid=abs(ir_recovery(x_plot,T1_final_grid,ra_final_grid,rb_final_grid))
plt.scatter(np.array(TIlist8000),TI_read_8000[2,:])
plt.legend(['Mz_Read'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.title(f'Simulation T1={T1} SNR={SNR}')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.text(x=8000,y=0,s=f'OLS:y={ra_final_ols:.02f}+{rb_final_ols:.02f}*e^-t/{int(T1_final_ols)}')
plt.text(x=8000,y=-.25,s=f'WLS:y={ra_final_wls:.02f}+{rb_final_wls:.02f}*e^-t/{int(T1_final_wls)}')
plt.text(x=8000,y=-0.5,s=f'Grid:y={ra_final_grid:.02f}+{rb_final_grid:.02f}*e^-t/{int(T1_final_grid)}')
plt.grid('True')
plt.scatter(np.array(TIlist[0:returnInd]),-1*np.abs(TI_read_8000[2,:][0:returnInd]),color='r')
plt.plot(x_plot,ydata_exp_ols,label='OLS')
plt.plot(x_plot,ydata_exp_wls,label='WLS')
plt.plot(x_plot,ydata_exp_grid,label='Grid')
plt.legend()
plt.show()
# %%
#Simulation:
#First, WLS + 8000 to get the T1
T1_simulate=np.arange(1000,1700,5)
TIlist1=[110,190,240,320,340,840,940,1240,1540,2140,2220,2740]
TIlist8000=[110,190,240,320,340,840,940,1240,1540,2140,2220,2740,8000]
SNR=20
errors=[]
T1final_list_WLS=[]
T1final_list_OLS=[]
T1final_list_Ori=[]
T1final_list_Grid=[]
T1final_list_WLS_8000=[]
T1final_list_OLS_8000=[]
T1final_list_Grid_8000=[]
for _ in range(1):
    for T1 in T1_simulate:
        time,M=simulateT1(dT=dT,T=T,df=df,T1=T1,T2=T2,Tr=Tr,flip_angle=flip_angle)
        M_noisy=add_noise(abs(M),SNR=SNR,type='Gaussian')
        TI_read=M_noisy[:,TIlist1]
        TI_read_8000=np.append(TI_read,M_noisy[:,40,np.newaxis],axis=-1)
        T1final_list_Ori.append(T1)
        print('Simulation to T1',T1)
        for searchtype in ['lm','grid']:
            if searchtype=='lm':
                for type in ['OLS','WLS']:
                    T1_final,ra_final,rb_final,res,returnInd=ir_fit(abs(TI_read[-1,:]),TIlist1,type=type,ra=1,rb=-2,T1=1200,Niter=2,searchtype=searchtype,error='l1')

                    if type =='OLS':
                        T1final_list_OLS.append(T1_final)
                    if type =='WLS':
                        T1final_list_WLS.append(T1_final)
            elif searchtype=='grid':
                T1_final,ra_final,rb_final,res,returnInd=ir_fit(abs(TI_read[-1,:]),TIlist1,type=type,searchtype=searchtype,T1bound=[1,5000])
                T1final_list_Grid.append(T1_final)
        for searchtype in ['lm','grid']:
            if searchtype=='lm':
                for type in ['OLS','WLS']:
                    T1_final,ra_final,rb_final,res,returnInd=ir_fit(abs(TI_read_8000[-1,:]),TIlist8000,type=type,Niter=2,searchtype=searchtype,error='l1')

                    if type =='OLS':
                        T1final_list_OLS_8000.append(T1_final)
                    if type =='WLS':
                        T1final_list_WLS_8000.append(T1_final)
            elif searchtype=='grid':
                T1_final,ra_final,rb_final,res,returnInd=ir_fit(abs(TI_read_8000[-1,:]),TIlist8000,type=type,searchtype=searchtype,T1bound=[1,5000])
                T1final_list_Grid_8000.append(T1_final)
savedict={}
savedict['T1final_list_WLS']=T1final_list_WLS
savedict['T1final_list_OLS']=T1final_list_OLS
savedict['T1final_list_Ori']=T1final_list_Ori
savedict['T1final_list_Grid']=T1final_list_Grid
savedict['T1final_list_WLS_8000']=T1final_list_WLS_8000
savedict['T1final_list_OLS_8000']=T1final_list_OLS_8000
savedict['T1final_list_Grid_8000']=T1final_list_Grid_8000
import pickle 
with open(f'saved_dictionary_{SNR}.pkl', 'wb') as f:
    pickle.dump(savedict, f)
# %%
plt.plot(T1final_list_Ori,T1final_list_WLS,label='WLS')
plt.plot(T1final_list_Ori,T1final_list_OLS,label='OLS')
plt.plot(T1final_list_Ori,T1final_list_Grid,label='Grid')
plt.legend()
plt.show()
plt.close()
# %%
plt.plot(T1final_list_Ori,T1final_list_WLS_8000,label='WLS_8000')
plt.plot(T1final_list_Ori,T1final_list_OLS_8000,label='OLS_8000')
plt.plot(T1final_list_Ori,T1final_list_Grid_8000,label='Grid_8000')
plt.legend()
plt.show()
plt.close()
# %%
plt.figure()
bland_altman_plot(T1final_list_Ori,T1final_list_OLS_8000,Print_title=f'Simulation Error SNR={SNR}\n  OLS_8000')
#plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} OLS'))
#plt.close()
plt.figure()
bland_altman_plot(T1final_list_Ori,T1final_list_WLS_8000,Print_title=f'Simulation Error SNR={SNR}\n  WLS_8000')
#plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} WLS'))
#plt.close()
plt.figure()
bland_altman_plot(T1final_list_Ori,T1final_list_Grid_8000,Print_title=f'Simulation Error SNR={SNR}\n  Grid_8000')
#plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} WLS'))
#plt.close()
# %%
plt.figure()
bland_altman_plot(T1final_list_Ori,T1final_list_OLS,Print_title=f'Simulation Error SNR={SNR}\n  OLS')
#plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} OLS'))
#plt.close()
plt.figure()
bland_altman_plot(T1final_list_Ori,T1final_list_WLS,Print_title=f'Simulation Error SNR={SNR}\n  WLS')
#plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} WLS'))
#plt.close()
plt.figure()
bland_altman_plot(T1final_list_Ori,T1final_list_Grid,Print_title=f'Simulation Error SNR={SNR}\n  Grid')
#plt.savefig(os.path.join(savedir,f'Simulation Error SNR={SNR} WLS'))
#plt.close()
# %%
