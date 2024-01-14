#%%
import numpy as np
import twixtools
import os
import matplotlib.pyplot as plt

image_default_root=r'C:\Research\MRI\MP_EPI\Phangtom'
image_dir=os.path.join(image_default_root,'CIRC_Phantom_Jan8 CIRC_Phantom_Jan8')
filename=os.path.join(image_dir,'meas_MID00762_FID22423_pulseq_se_dwi_x')
multi_twix = twixtools.read_twix(filename)

def read_image_data(filename):
    out = []
    for mdb in twixtools.read_twix(filename)[-1]['mdb']:
        if mdb.is_image_scan():
            out.append(mdb.data)
    return np.asarray(out)  # 3D numpy array [acquisition_counter, n_channel, n_column]

# %%
#For the SE_DWI: it's [acquisition_counter, n_channel, n_column]
#It's 32*n, n_channel, n_column
#Therefore:
Nx=64
Ny=64
mdb_data=read_image_data(filename)
Nx,Nchannel,Ncolumn=np.shape(mdb_data)
b_all=np.zeros((Nx,Nchannel,Ncolumn,6),dtype=complex)
list=[]
for dirpath,basenames,filesname in os.walk(image_dir):
    for x in filesname:
        path=os.path.join(dirpath,x)
        if  'se_dwi' in path.lower():
            list.append(path)
print(list)
#b_500_n_channel=mdb_data[Nx::,:,:]

for ind,path in enumerate(list[0:6]):
    b_all[:,:,:,ind]=read_image_data(path)

print(np.shape(b_all))

#%%
plt.figure(figsize=(6*3,3*2))
b_images=np.zeros((Nx,Ncolumn,6),dtype=complex)
b_images_maptitute=np.zeros((Nx,Ncolumn,6))

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
np.save(os.path.join(image_dir,'DWI_6'),b_images_maptitute)

print(np.shape(b_images_maptitute))
# %%
