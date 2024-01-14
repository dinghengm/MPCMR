import numpy as np
import numpy.matlib 
import scipy as sp 
import scipy.io as io
import matplotlib.pyplot as pl
import nibabel as nib 
import time
from ipyfilechooser import FileChooser
import matplotlib.pyplot as plt
import os
def imshowMap(obj,path,plot):
    num_slice=obj.Nz
    volume=obj._map
    ID=str('map_' + obj.CIRC_ID + '_' + obj.ID)
    crange=obj.crange
    cmap=obj.cmap
    figsize = (3.4*num_slice, 3)

    fig, axes = plt.subplots(nrows=1, ncols=num_slice, figsize=figsize, constrained_layout=True)
    axes=axes.ravel()
    for sl in range(num_slice):
        axes[sl].set_axis_off()
        im = axes[sl].imshow(volume[..., sl],vmin=crange[0],vmax=crange[1], cmap=cmap)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.018, aspect=8)
    img_dir= os.path.join(path,f'{ID}')
    if plot:
        plt.savefig(img_dir)
    pass
# conda install -c conda-forge ipyfilechooser


# helper functions
def toc(t_start): #returns elapsted time given start time t
    tElapsed = time.time() - t_start
    if (tElapsed / 3600 > 1):
        return "{:.4f}".format(tElapsed/3600) + " hours"
    elif (tElapsed / 60 > 1):
        return "{:.4f}".format(tElapsed/60) + " mins"
    else:
        return "{:.4f}".format(tElapsed) +" secs"

def imshowgrid(images,titles=[],NCol=3):
    if type(images) != tuple:
        raise Exception("Needs to be tuple of images")

    NImages = len(images)
    
    if len(titles) == 0:
        titles = np.arange(NImages).astype(str)

    fig, ax = pl.subplots(int(np.ceil(NImages/NCol)),np.min([NImages,NCol]), constrained_layout=True)
    for i, image in enumerate(images):
        ax[i].imshow(np.abs(image), origin='lower', cmap='gray')
        ax[i].set_title(titles[i])

def phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
	"""
	 phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)
	
	Create a Shepp-Logan or modified Shepp-Logan phantom.

	A phantom is a known object (either real or purely mathematical) 
	that is used for testing image reconstruction algorithms.  The 
	Shepp-Logan phantom is a popular mathematical model of a cranial
	slice, made up of a set of ellipses.  This allows rigorous 
	testing of computed tomography (CT) algorithms as it can be 
	analytically transformed with the radon transform (see the 
	function `radon').
	
	Inputs
	------
	n : The edge length of the square image to be produced.
	
	p_type : The type of phantom to produce. Either 
	  "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
	  if `ellipses' is also specified.
	
	ellipses : Custom set of ellipses to use.  These should be in 
	  the form
	  	[[I, a, b, x0, y0, phi],
	  	 [I, a, b, x0, y0, phi],
	  	 ...]
	  where each row defines an ellipse.
	  I : Additive intensity of the ellipse.
	  a : Length of the major axis.
	  b : Length of the minor axis.
	  x0 : Horizontal offset of the centre of the ellipse.
	  y0 : Vertical offset of the centre of the ellipse.
	  phi : Counterclockwise rotation of the ellipse in degrees,
	        measured as the angle between the horizontal axis and 
	        the ellipse major axis.
	  The image bounding box in the algorithm is [-1, -1], [1, 1], 
	  so the values of a, b, x0, y0 should all be specified with
	  respect to this box.
	
	Output
	------
	P : A phantom image.
	
	Usage example
	-------------
	  import matplotlib.pyplot as pl
	  P = phantom ()
	  pl.imshow (P)
	
	References
	----------
	Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue 
	from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
	Feb. 1974, p. 232.
	
	Toft, P.; "The Radon Transform - Theory and Implementation", 
	Ph.D. thesis, Department of Mathematical Modelling, Technical 
	University of Denmark, June 1996.
	
	"""
	
	if (ellipses is None):
		ellipses = _select_phantom (p_type)
	elif (np.size (ellipses, 1) != 6):
		raise AssertionError ("Wrong number of columns in user phantom")
	
	# Blank image
	p = np.zeros ((n, n))

	# Create the pixel grid
	ygrid, xgrid = np.mgrid[-1:1:(1j*n), -1:1:(1j*n)]

	for ellip in ellipses:
		I   = ellip [0]
		a2  = ellip [1]**2
		b2  = ellip [2]**2
		x0  = ellip [3]
		y0  = ellip [4]
		phi = ellip [5] * np.pi / 180  # Rotation angle in radians
		
		# Create the offset x and y values for the grid
		x = xgrid - x0
		y = ygrid - y0
		
		cos_p = np.cos (phi) 
		sin_p = np.sin (phi)
		
		# Find the pixels within the ellipse
		locs = (((x * cos_p + y * sin_p)**2) / a2 
              + ((y * cos_p - x * sin_p)**2) / b2) <= 1
		
		# Add the ellipse intensity to those pixels
		p [locs] += I

	return p


def _select_phantom (name):
	if (name.lower () == 'shepp-logan'):
		e = _shepp_logan ()
	elif (name.lower () == 'modified shepp-logan'):
		e = _mod_shepp_logan ()
	else:
		raise ValueError ("Unknown phantom type: %s" % name)
	
	return e


def _shepp_logan ():
	#  Standard head phantom, taken from Shepp & Logan
	return [[   2,   .69,   .92,    0,      0,   0],
	        [-.98, .6624, .8740,    0, -.0184,   0],
	        [-.02, .1100, .3100,  .22,      0, -18],
	        [-.02, .1600, .4100, -.22,      0,  18],
	        [ .01, .2100, .2500,    0,    .35,   0],
	        [ .01, .0460, .0460,    0,     .1,   0],
	        [ .02, .0460, .0460,    0,    -.1,   0],
	        [ .01, .0460, .0230, -.08,  -.605,   0],
	        [ .01, .0230, .0230,    0,  -.606,   0],
	        [ .01, .0230, .0460,  .06,  -.605,   0]]

def _mod_shepp_logan ():
	#  Modified version of Shepp & Logan's head phantom, 
	#  adjusted to improve contrast.  Taken from Toft.
	return [[   1,   .69,   .92,    0,      0,   0],
	        [-.80, .6624, .8740,    0, -.0184,   0],
	        [-.20, .1100, .3100,  .22,      0, -18],
	        [-.20, .1600, .4100, -.22,      0,  18],
	        [ .10, .2100, .2500,    0,    .35,   0],
	        [ .10, .0460, .0460,    0,     .1,   0],
	        [ .10, .0460, .0460,    0,    -.1,   0],
	        [ .10, .0460, .0230, -.08,  -.605,   0],
	        [ .10, .0230, .0230,    0,  -.606,   0],
	        [ .10, .0230, .0460,  .06,  -.605,   0]]


def uigetfile(path=None, 
            pattern=['*.jpg', '*.jpeg','*.png', '*.dat', '*.nii', '*.nii.gz', '*.mat', '*.gif'],
            callbackFunc=None):
    # Create and display a FileChooser widget
    fc = FileChooser('.')
    display(fc)
    
    # Print the selected path, filename, or both
    #print(fc.selected_path)
    #print(fc.selected_filename)
    #print(fc.selected)

    # Change defaults and reset the dialog
    if path == None:
        path = '.'
    
    fc.default_path = path # 
    fc.default_filename = 'test.dat'
    fc.reset()

    # Shorthand reset
    fc.reset(path=fc.default_path, filename='output.txt')

    # Change hidden files
    fc.show_hidden = True

    # Show or hide folder icons
    fc.use_dir_icons = True

    # Switch to folder-only mode
    #fc.show_only_dirs = True

    # Set multiple file filter patterns (uses https://docs.python.org/3/library/fnmatch.html)
    fc.filter_pattern = pattern

    # Change the title (use '' to hide)
    fc.title = '<b> Choose a data file (*.jpeg, *.png, *.mat, *.gif, *.dat, *.nii, *.nii.gz) </b>'

    # Sample callback function
    def change_title(chooser):
        #fc.title = '<b>Callback function executed</b>'
        filename = fc.selected_path +'/'+ fc.selected_filename
        print(filename)
        
    # Register callback function
    if callbackFunc == None:
        fc.register_callback(change_title)
    else:
        fc.register_callback(callbackFunc)

    return fc
