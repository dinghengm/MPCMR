#%%

##############If you only wants to generate the np file, just run first ,4 and the rest cells#################
import numpy as np
import twixtools
import os
import matplotlib.pyplot as plt

image_default_root=r'C:\Research\MRI\MP_EPI\Phantom'
#NO3
image_dir=os.path.join(image_default_root,'CIRC_Phantom_Jan22_Final CIRC_Phantom_Jan22_Final','SE_DWI_1')

#%%
filename=os.path.join(image_dir,'meas_MID00364_FID25067_external_b50_3')
multi_twix = twixtools.read_twix(filename)
def read_image_data(filename):
    out = []
    for mdb in twixtools.read_twix(filename)[-1]['mdb']:
        if mdb.is_image_scan():
            out.append(mdb.data)
    return np.asarray(out)  # 3D numpy array [acquisition_counter, n_channel, n_column]
Nx=160
Ny=64
mdb_data=read_image_data(filename)
Nx,Nchannel,Ncolumn=np.shape(mdb_data)
b1=np.mean(mdb_data[0:64,:,:],axis=1)
#b_complex=np.mean(mdb_data,axis=1)
b_complex=b1
plt.subplot(2,1,1)
plt.imshow(np.log(abs(b_complex)+ 1e-9),cmap='gray')
plt.title('ksapce')
plt.subplot(2,1,2)
b_img=np.fft.ifft2(np.fft.ifftshift(b_complex))
b_img=np.fft.fftshift(b_img)
plt.title('image')
plt.axis('off')
plt.imshow(abs(b_img),cmap='gray')



# %%
#For the SE_DWI: it's [acquisition_counter, n_channel, n_column]
#It's 32*n, n_channel, n_column
#Therefore:
Nx=160
Ny=64
mdb_data=read_image_data(filename)
Nx,Nchannel,Ncolumn=np.shape(mdb_data)
b_all=np.zeros((Ny,Nchannel,Ncolumn,6),dtype=complex)
#%%
list=[]
for dirpath,basenames,filesname in os.walk(image_dir):
    for x in filesname:
        path=os.path.join(dirpath,x)
        if  'external' in path.lower():
            list.append(path)
print(list)
#b_500_n_channel=mdb_data[Nx::,:,:]
k=0
for ind,path in enumerate(list[0:2]):
    k=0
    for i in range(3):
        b_all[:,:,:,i+ind*3]=read_image_data(path)[k:k+64,:,:]

print(np.shape(b_all))

#%%
plt.figure(figsize=(6*3,3*2))
b_images=np.zeros((Ny,Ncolumn,6),dtype=complex)
b_images_maptitute=np.zeros((Ny,Ncolumn,6))

for i in range(6):
    b_complex=np.mean(b_all[:,:,:,i],axis=1)
    plt.subplot(2,6,i+1)
    plt.imshow(np.log(abs(b_complex)+ 1e-9),cmap='gray')
    plt.title('ksapce')
    plt.subplot(2,6,i+7)
    b_img=np.fft.ifft2(np.fft.ifftshift(b_complex))
    b_img=np.fft.fftshift(b_img)
    b_images[:,:,i]= b_img
    plt.title('image')
    plt.axis('off')
    b_images_maptitute[:,:,i]=np.abs(b_img)
    plt.imshow(abs(b_img),cmap='gray')


#%%
np.save(os.path.join(image_dir,'DWI_6_Jan22_b50_1'),b_images_maptitute)

print(np.shape(b_images_maptitute))
# %%
