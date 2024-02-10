#%%
import twixtools
import numpy as np
import matplotlib.pyplot as plt
import os


def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig)**2, axis))

#%%
image_default_root=r'C:\Research\MRI\MP_EPI\Phantom'
image_dir=os.path.join(image_default_root,'CIRC_Phantom_Feb_9_Diff_SE_EPI')
filename=os.path.join(image_dir,'meas_MID01537_FID27999_ep2d_MP03_DWI_Z.dat')
epi_map = twixtools.map_twix(filename)

im_array = epi_map[-1]['image']
print(im_array.non_singleton_dims)
print(np.shape(im_array))
im_array.flags['remove_os'] = True

#%%
# read the data  # It's 
#['Rep', 'Sli', 'Lin', 'Cha', 'Col']

data = im_array[:].squeeze()
image_50_k=data[0,0,:,:,:]
image_500_k=data[3,0,:,:,:]
#There fore we have to use 0-5 for different repetition

# reconstruct the data
image_50 = ifftnd(image_50_k, [0,-1])
image_50 = rms_comb(image_50)

# plot the data

plt.figure(figsize=(12,6))
plt.subplot(221)
plt.title('k-space')
plt.imshow(np.log(abs(np.mean(image_50_k,axis=1))+ 1e-9), cmap='gray', origin='lower')
plt.axis('off')

plt.subplot(223)
plt.title('image_50')
plt.imshow(image_50, cmap='gray', origin='lower')
plt.axis('off')

image_500 = ifftnd(image_500_k, [0,-1])
image_500 = rms_comb(image_500)

# plot the data

plt.subplot(222)
plt.title('k-space')
plt.imshow(np.log(abs(np.mean(image_500_k,axis=1))+ 1e-9), cmap='gray', origin='lower')
plt.axis('off')

plt.subplot(224)
plt.title('image_500')
plt.imshow(image_500, cmap='gray', origin='lower')
plt.axis('off')

# %%
# functions to calculate & apply phase-correction:

def calc_pc_corr(sig):
    ncol = sig.shape[-1]
    
    # ifft col dim.
    pc = ifftnd(sig, [-1])
    
    # calculate phase slope from autocorrelation (for both readout polarities separately - each in its own dim)
    slope = np.angle((np.conj(pc[...,1:]) * pc[...,:-1]).sum(-1, keepdims=True).sum(-2, keepdims=True))
    x = np.arange(ncol) - ncol//2
    
    return np.exp(1j * slope * x)


def apply_pc_corr(sig, pc_corr):
    
    # ifft col dim.
    sig = ifftnd(sig, [-1])
    
    # apply phase-correction slope
    sig *= pc_corr
    
    # remove segment dim
    sig = sig.sum(5).squeeze()
    
    # ifft lin dim.
    sig = fftnd(sig, [-1])
    
    return sig


# for phase-correction, we need to keep the individual segments (which indicate the readout's polarity)
im_array.flags['remove_os'] = True
im_array.flags['average']['Seg'] = False

pc_array = epi_map[-1]['phasecorr']
pc_array.flags['remove_os'] = True
pc_array.flags['skip_empty_lead']=True
pc_array.flags['average']['Seg'] = False

# calculate phase-correction
pc_corr = calc_pc_corr(pc_array[:])

# apply phase-correction
image_pc = apply_pc_corr(im_array[:], pc_corr)

# RMS coil combination# plot results

#image_pc = rms_comb(image_pc)

#%%
print(np.shape(image_pc))

#%%
# read the data  # It's 
#['Rep', 'Sli', 'Lin', 'Cha', 'Col']

data_pc = image_pc[:].squeeze()
image_50_k=data_pc[0,2,:,:,:]
image_500_k=data_pc[3,2,:,:,:]
#There fore we have to use 0-5 for different repetition

# reconstruct the data
image_50 = ifftnd(image_50_k, [0,-1])
image_50 = rms_comb(image_50)

# plot the data

plt.figure(figsize=(12,6))
plt.subplot(221)
plt.title('k-space')
plt.imshow(np.log(abs(np.mean(image_50_k,axis=1))+ 1e-9), cmap='gray', origin='lower')
plt.axis('off')

plt.subplot(223)
plt.title('image_50')
plt.imshow(image_50, cmap='gray', origin='lower')
plt.axis('off')

image_500 = ifftnd(image_500_k, [0,-1])
image_500 = rms_comb(image_500)

# plot the data

plt.subplot(222)
plt.title('k-space')
plt.imshow(np.log(abs(np.mean(image_500_k,axis=1))+ 1e-9), cmap='gray', origin='lower')
plt.axis('off')

plt.subplot(224)
plt.title('image_500')
plt.imshow(image_500, cmap='gray', origin='lower')
plt.axis('off')

# %%
