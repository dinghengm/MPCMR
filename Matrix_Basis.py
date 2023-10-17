#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy

# %%
#Relaxation
def traRelax(T2,time):
    '''
    Transverse relaxation
    input: Matrix(3*3),T2,time
    output:rotation Matrix A
    ''' 
    A=np.eye(3)
    A[0]=A[0]*np.exp(-time/T2)
    A[1]=A[1]*np.exp(-time/T2)
    return A
def longRelax(T1,time,M0=np.array([0,0,1])):
    '''
    Transverse relaxation
    input: M,T2,time,M0(default)[0,0,1]
    output:Rotation matrix A, and vectorB
    '''  
    A=np.eye(3) 
    A[2]=A[2]*np.exp(-time/T1)
    B=M0-M0*np.exp(-time/T1)
    return A,B
def Relaxation(T1,T2,time):
    '''
    Transverse relaxation
    input: M,T2,time,M0(default)[0,0,1]
    output:A,B
    ''' 
    A,B=longRelax(T1,time)
    traA=traRelax(T2,time)
    A[0]=traA[0]
    A[1]=traA[1]
    return A,B


#Precession
def zrot(phi):
    '''
    Rotate among z axis
    input: degree in radial form
    output:Rotation matrix
    '''
    Rz=[[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]]
    return Rz
def xrot(phi):
    '''
    Rotate among x axis
    input: degree in radial form
    output:Rotation matrix
    '''
    Rx=[[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]]
    return Rx
def yrot(phi):
    '''
    Rotate among y axis
    input: degree in radial form
    output:Rotation matrix
    '''
    Ry=[[np.cos(phi),0,np.sin(phi)],[0,1,0],[-np.sin(phi),0,np.cos(phi)]]
    return Ry
def throt(phi,theta):
    '''
    Rotate among a transverse plane, with y=x*tan(theta)
    input: degree in radial form
    output:Rotation matrix
    '''
    Rz = zrot(-theta)
    Rx = xrot(phi)
    Rth = np.dot(Rx,Rz)
    Rth=np.dot(np.linalg.inv(Rz),Rth)
    return Rth

def freePrecess(T,T1,T2,df):
    '''
    Function simulates free precession and decay
    input:time interval T, T1,T2 and off resoncance in Hz df
    output:Rotation matrix Afp,Bfp
        
    '''

    phi = 2*np.pi *df * T /1000
    E1=np.exp(-T/T1)
    E2=np.exp(-T/T2)
    Afp=np.array([[E2,0,0],[0,E2,0],[0,0,E1]])
    Afp=np.dot(Afp,zrot(phi))
    Bfp=np.array(np.transpose([0,0,1-E1]))
    return Afp,Bfp

def ssSignal(flip,T1,T2,TE,TR,df):
    '''
    Function simulation about the steady state
    input:flip angle, T1,T2,TE,TRand off resoncance in Hz df
    output:Rotation matrix Afp,Bfp
        
    '''
    Atr,Btr=freePrecess(TR,T1,T2,df)
    Ate,Bte=freePrecess(TE,T1,T2,df)
    Ry=yrot(flip)
    M=np.dot(np.linalg.inv(np.eye(3)-np.dot(Atr,Ry)),Btr)
    M1=np.dot(M,Ry)
    output=np.dot(Ate,M1)+Bte
    return output


def srSignal(flip,T1,T2,TE,TR,df=0):
    '''
    Function simulation about the steady state but multipy Atr by [0,0,0][0,0,0][0,0,1]
    Null all transverse magnetization
    input:flip angle, T1,T2,TE,TRand off resoncance in Hz df
    output:Rotation matrix Afp,Bfp
    '''
    Atr,Btr=freePrecess(TR-TE,T1,T2,df)
    Ate,Bte=freePrecess(TE,T1,T2,df)
    Atr=np.dot(np.array([[0,0,0],[0,0,0],[0,0,1]]),Atr)
    Rflip=yrot(flip)
    Mss = np.linalg.inv(np.eye(3)-Ate@Rflip@Atr) @ (Ate@Rflip@Btr+Bte);
    return Mss    

def seSignal(T1,T2,TE,TR,df):
    '''
    Spin Echo at steady state but multipy Atr by [0,0,0][0,0,0][0,0,1]
    Null all transverse magnetization
    input:T1,T2,TE,TRand off resoncance in Hz df
    output:Rotation matrix Afp,Bfp
    '''
    Atr,Btr=freePrecess(TR-TE,T1,T2,df)
    Ate2,Bte2=freePrecess(TE/2,T1,T2,df)
    Atr=np.dot(np.array([[0,0,0],[0,0,0],[0,0,1]]),Atr)
    Rflip=yrot(np.pi/2)
    Rrefoc=xrot(np.pi)
    Mss = np.linalg.inv(np.eye(3)-Ate2@Rrefoc@Ate2@Rflip@Atr) @ (Bte2+Ate2@Rrefoc@(Bte2+Ate2@Rflip@Btr))
    return Mss


def fsesignal(T1,T2,TE,TR,ETL,df=0):
    '''
    --------------It's Blank!!!
    Fast Spin Echo at steady state but multipy Atr by [0,0,0][0,0,0][0,0,1]
    Null all transverse magnetization
    input:T1,T2,TE,TRand off resoncance in Hz df
    output:Rotation matrix Afp,Bfp
    '''
    Atr,Btr=freePrecess(TR-TE,T1,T2,df)
    Ate2,Bte2=freePrecess(TE/2,T1,T2,df)
    Rflip=yrot(np.pi/2)
    Rrefoc=xrot(np.pi)
    pass



def gsSignal(flip,T1,T2,TE,TR,df,phi):
    '''
    Calculate the steady state signal at TE for repeated
    Null all transverse magnetization
    input:flip angle, T1,T2,TE,TRand off resoncance in Hz df
    output:Rotation matrix Afp,Bfp
    '''
    Atr,Btr=freePrecess(TR-TE,T1,T2,df)
    Ate,Bte=freePrecess(TE,T1,T2,df)
    Rflip=yrot(flip)
    Atr=zrot(phi)@Atr
    Mss=np.linalg.inv(np.eye(3)-Ate@Rflip@Atr) @ (Ate@Rflip@Btr+Bte)

    return Mss   

def greSignal(flip,T1,T2,TE,TR,dfreq):
    N=100
    M=np.zeros((3,N))
    phi = (np.arange(100)/N-0.5 )*4*np.pi


    for k in range(N):
        M1=gsSignal(flip,T1,T2,TE,TR,dfreq,phi[k])
        M[:,k]=M1[:]

    Mss = np.mean(M,axis=1)
    return Mss



# %%
