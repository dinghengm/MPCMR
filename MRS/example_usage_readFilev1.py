#%%
import argparse
import sys
sys.path.append('../MPEPI')
from libMapping_v13 import mapping,readFolder  # <--- this is all you need to do diffusion processing
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from tqdm.auto import tqdm # progress bar
import pandas as pd
import warnings #we know deprecation may show bc we are using a stable older ITK version
defaultPath= r'C:\Research\MRI\MRS'
# %%
from libDiffusion_DCK import *

a=diffusion(r'C:\Research\MRI\MP_EPI\Diffusion\CIRC_00381_22737.diffusion')

# %%
np.bool=bool
np.int=int
np.float=float

# %%
