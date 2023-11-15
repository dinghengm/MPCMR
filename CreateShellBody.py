# %% load required programs
# for shell creation
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from scipy.ndimage import sobel
from scipy.ndimage import generic_gradient_magnitude
import nibabel as nib
#from viz3d import viz3d

# to save dcm
import datetime
import os
import tempfile
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID
import tqdm
#import dicom2nifti

# %% load file and set image, visualize in 3D
ds=nib.load(r'C:\Research\MRI\exvivo\UCSF_7020_05162023\UCSF_7020_05162023.nii.gz')
image=ds.get_fdata()[:,:,:,0]

#ds = pydicom.read_file('/Users/ROBAKOM/Library/CloudStorage/OneDrive-ClevelandClinic/Segmentation/Shell 35/_Raw MRI/Shell 35 MRI.dcm')
#image = ds.pixel_array
print(ds)
#%%
x = 50     # slice being looked at
#viz3d(image,axis="x")
plt.imshow(image[x],cmap="gray")

# %% rotate and remove body - transverse

imx = np.rot90(image, k=1, axes=(1,2))      # sagittal

startx = 8
stopx = 68

imagex = imx[startx:stopx]
#%%
#viz3d(imagex,axis="x")
y = x - startx
plt.imshow(imagex[y],cmap="gray")

# %% rotate and remove body - saggital

imy = np.rot90(imagex, k=1, axes=(0,2))      # coronal

starty = 10
stopy = 66

imagey = imy[starty:stopy]

#viz3d(imagey,axis="x")
plt.imshow(imagey[y],cmap="gray")

# %% rotate and remove body - coronal

imz = np.rot90(imagey, k=1, axes=(1,0))      # transverse

startz = 10
stopz = 66

imagez = imz[startz:stopz]

#viz3d(imagez,axis="x")
plt.imshow(imagez[y],cmap="gray")
#%%
image=image[startx:stopx,starty:stopy,startz:stopz]



# %% edge detection

image_edge = generic_gradient_magnitude(image,sobel)

#viz3d(image_edge,axis="x")
plt.imshow(image_edge[y],cmap="gray")

# %% apply threshhold and revisualize
image_th1 = np.zeros(image.shape)
for i in range(image.shape[0]):
    thresh_l = np.max(image[i]) * 0.3     # changes
    #thresh_h= np.max(image[i]) * 0.02
    image_th1[image < thresh_l] =1
    #image_th1[image < thresh_h] =0

#viz3d(image_th1,axis="x")
plt.imshow(image_th1[y],cmap="gray")

#newimage = np.rot90(image_th1, k=1, axes=(0,2))
#plt.imshow(newimage[y],cmap="gray")

# %% blur image and revisualize
from scipy.ndimage import gaussian_filter

sig = 0.8 # changes
image_blr = gaussian_filter(image_th1,sigma=[sig,sig,sig])

#viz3d(image_blr,axis="x")
plt.imshow(image_blr[y],cmap="gray")

# %% reapply threshhold and revisualize
image_th2 = np.zeros(image_th1.shape)
image_th2[image_blr < 0.8] = 1       # changes

#viz3d(image_th2,axis="x")
plt.imshow(image_th2[y],cmap="gray")

# %% edge detection

image_edge2 = generic_gradient_magnitude(image_th2,sobel)

#viz3d(image_edge,axis="x")
plt.imshow(image_edge2[y],cmap="gray")

# %% reapply threshold
shell = np.zeros(image_edge2.shape)
shell[image_edge2 < 0.4*np.max(image_edge2)] = 1        # changes

#viz3d(shell,axis="x")
plt.imshow(shell[y],cmap="gray")

# %% save file as nifti array

shell_nif = nib.Nifti1Image(np.flip(shell, axis=0), affine=np.eye(4))
nib.save(shell_nif, "/Users/ROBAKOM/Library/CloudStorage/OneDrive-ClevelandClinic/Segmentation/Shell 35/_Raw MRI/Shell 35_All")       # changes

# %% save as dicom series

# set up

# file name
data_nifti = "Shell 35_All.nii"        # changes

im3D = np.array((nib.load(data_nifti)).dataobj)
im3D = 255 * im3D / np.max(im3D)
im3D = im3D.astype(np.uint16)

# directory location
dcm_dir = "/Users/ROBAKOM/Library/CloudStorage/OneDrive-ClevelandClinic/Segmentation/Shell 35/_Raw MRI/Shell_All"   # change

# set meta info

#print("Setting file meta information...")
# populate required values for file meta information
file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')
file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
file_meta.ImplementationClassUID = UID("1.2.3.4")

# loop over 3D data

for nz in range(im3D.shape[0]):

    # Create filename
    filename = os.path.join(dcm_dir, f"Shell 35_Slice{nz}.dcm")       # changes

    #print("Setting dataset values...")
    # Create the FileDataset instance (initially no data elements, but file_meta
    # supplied)
    ds = FileDataset(filename, {}, file_meta = file_meta, preamble=b"\0" * 128)

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "Porcine"        # changes
    ds.PatientID = "00000"
    ds.Modality = "MR"

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelRepresentation = 1

    ds.Rows = im3D.shape[1]
    ds.Columns = im3D.shape[2]

    ds.AcquisitionNumber = 1
    ds.InstanceNumber = nz + 1

    ds.PixelData = im3D[nz,:,:].tobytes()

    # Set the transfer syntax
    # Write as a different transfer syntax XXX shouldn't need this but pydicom
    # 0.9.5 bug not recognizing transfer syntax
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
    ds.is_little_endian = False
    ds.is_implicit_VR = False

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr

    #print("Writing test file as Big Endian Explicit VR", filename)
    ds.save_as(filename)

# %% safe as series of png images

for j in range(shell.shape[0]):
     plt.imshow(shell[j],cmap="gray")
     plt.axis("off")
     plt.tight_layout()
     plt.savefig(f'Shell_Images/Shell_35_png/Shell_35_Slice{j}.png', bbox_inches="tight", pad_inches=0)   #changes

# %% check png image dimensions

for k in range(shell.shape[0]):
    img = Image.open(f'Shell_Images/Shell_35_png/Shell_35_Slice{k}.png')    # changes
    # get width and height
    width = img.width
    height = img.height
   
    # display width and height
    print(f"The height of slice {k} is: ", height)
    print(f"The width of slice {k} is: ", width)
# %%
newimage = np.rot90(image_th1, k=1, axes=(0,2))
plt.imshow(newimage[y],cmap="gray")
# %%
