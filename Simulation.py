#%%
import random
from Matrix_Basis import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
# %%
dT=1       #1 ms
T=6000   #Total Duration 1s
df=0        #10Hz in second
T1=1300
T2=40
Te=40
Tr=6000
N=int(T/dT)+1  #number of time steps
#M is the 3 * time
M=np.zeros((3,N))
A,B = freePrecess(dT,T1,T2,df)
Ate,Bte=freePrecess(Te,T1,T2,df)
Atr,Btr=freePrecess(Tr,T1,T2,df)
Ry=yrot(np.pi)
M[:,0]=np.array(np.transpose([0,0,1]))
M[:,0]=np.dot(M[:,0],Ry)
for k in range(1,N,1):
    M[:,k]=np.dot(A,M[:,k-1])+B
    if (k % Tr==0):
    #M[:,k]=np.dot(A,M[:,k-1])+B
        M[:,k]=np.dot(M[:,k-1],Ry)
time=np.arange(0,N,1)*dT
#plt.plot(time,np.abs(M[0,:]),'b-')
#plt.plot(time,np.abs(M[1,:]),'r--')
plt.plot(time,np.abs(M[2,:]),'g-.')
plt.legend(['Mz'])
plt.xlabel('Time (ms)')
plt.ylabel('Magnetization')
plt.axis(xmin=np.min(time),xmax=np.max(time),ymin=-1,ymax=1)
plt.grid('True')
plt.show()
plt.close()
# %%

#Add noise to the Mz
