# %% ####################################################################################
# Import libraries ######################################################################
#########################################################################################
# all you need is below (must have the matplotlib qt for GUI like crop or lv segmentation)
%matplotlib qt                      
from libMapping_v12 import mapping  # <--- this is all you need to do diffusion processing
from libMapping_v12 import readFolder
import numpy as np
import matplotlib.pyplot as plt
import os


import warnings #we know deprecation may show bc we are using a stable older ITK version
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# %%
dicomPath=r'C:\Research\MRI\MP_EPI\CIRC_00302_22737_CIRC_00302_22737\MP03_DWI_Zoom\MR000000.dcm'
CIRC_ID='CIRC_00302'
ID = os.path.dirname(dicomPath).split('\\')[-1]
MP03 = mapping(data=dicomPath,ID=ID,CIRC_ID=CIRC_ID)
# %%
#Motion correction
MP03.go_crop()
MP03.go_resize(scale=2)
MP03.go_moco()
MP03.imshow_corrected()
#%%
#Get the Maps
ADC=MP03.go_cal_ADC()
MP03._map=ADC*1000
MP03.imshow_map()
# %%

#%matplotlib qt <--- you need this if you haven't turned it on in vscode
MP03.go_segment_LV(reject=None, image_type="b0")

# save
MP03.save()

# look at stats
MP03.show_calc_stats_LV()

MP03.imshow_overlay()
# %%
