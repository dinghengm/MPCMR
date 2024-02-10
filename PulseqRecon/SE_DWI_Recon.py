#%%
import numpy as np
import twixtools
import os
import matplotlib.pyplot as plt

image_default_root=r'C:\Research\MRI\MP_EPI\Phantom'
image_dir=os.path.join(image_default_root,'CIRC_Phantom_Dec_21_DWI_SE')
filename=os.path.join(image_dir,'meas_MID00087_FID20305_pulseq_SE_DWI')
multi_twix = twixtools.read_twix(filename)

def read_image_data(filename):
    out = list()
    for mdb in twixtools.read_twix(filename)[-1]['mdb']:
        if mdb.is_image_scan():
            out.append(mdb.data)
    return np.asarray(out)  # 3D numpy array [acquisition_counter, n_channel, n_column]
#%%
mdb = multi_twix[-1]['mdb'][0] # first mdb element of last measurement
mdb.data # data of first mdb (may or may not be imaging data)
mdb.mdh
# %%
#For the SE_DWI: it's [acquisition_counter, n_channel, n_column]
#It's 32*n, n_channel, n_column
#Therefore:
Nx=32
Ny=32



mdb_data=read_image_data(filename)
b_50_n_channel=mdb_data[0:Nx,:,:]
b_500_n_channel=mdb_data[Nx::,:,:]

#%%
#Plot average
b_50_complex=np.mean(b_50_n_channel,axis=1)
b_500_complex=np.mean(b_500_n_channel,axis=1)

plt.figure(figsize=(12,6))
plt.subplot(221)
plt.imshow(np.log(abs(b_50_complex)+ 1e-9),cmap='gray')
plt.title('ksapce-b50')
plt.subplot(222)
plt.imshow(np.log(abs(b_500_complex)+ 1e-9),cmap='gray')
plt.title('ksapce-b500')
plt.subplot(223)
b_50=np.fft.ifft2(np.fft.ifftshift(b_50_complex))
b_50=np.fft.fftshift(b_50)
plt.axis('off')
plt.imshow(abs(b_50),cmap='gray')
plt.title('b50')
plt.subplot(224)
plt.axis('off')
b_500=np.fft.ifft2(np.fft.ifftshift(b_500_complex))
b_500=np.fft.fftshift(b_500)
plt.title('b500')
plt.imshow(abs(b_500),cmap='gray')



# %%
