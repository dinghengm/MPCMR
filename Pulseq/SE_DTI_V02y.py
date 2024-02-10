#%%

import math

import numpy as np

import pypulseq as pp



#Very slow step to test for TE, TR or within slewrate limits
def bFactCalc(g,delta,DELTA):
    ''' see DAVY SINNAEVE Concepts in Magnetic Resonance Part A, Vol. 40A(2) 39–65 (2012) DOI 10.1002/cmr.a
    b = gamma^2  g^2 delta^2 sigma^2 (DELTA + 2 (kappa - lambda) delta)
    in pulseq we don't need gamma as our gradinets are Hz/m
    however, we do need 2pi as diffusion equations are all based on phase
    for rect gradients: sigma=1 lambda=1/2 kappa=1/3
    '''
    sigma =1
    kappa_minus_lambda=1/3-1/2
    b=(2*np.pi *g * delta*sigma)**2 * (DELTA + 2 * kappa_minus_lambda*delta)
    return b


#%%

plot=bool
write_seq=True
seq_filename="se_dwi_pypulseq_TE65_FOV172_Nx64_y.seq"

# ======
# SETUP
# ======
seq = pp.Sequence()  # Create a new sequence object
fov = 172e-3  # Define FOV and resolution
Nx = 64
Ny = 64
slice_thickness=8e-3
Nslices=1
bFactor50=50
bFactor500=500
TE=60e-3
TR=8        #8s

# Set system limits
system = pp.Opts(
    max_grad=80,
    grad_unit="mT/m",
    max_slew=180,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,   #default
    rf_dead_time=100e-6,    #default
    adc_dead_time=20e-6,    #default
)
# ======
# CREATE EVENTS
# ======
# Create 90 degree slice selection pulse and gradient
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=3e-3,
    #Slice thickness is 8e-3
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
)

# Define other gradients and ADC events
delta_k = 1 / fov
k_width = Nx * delta_k
readout_time=3.2e-4   #TODO #Might be able to change to match bandwidth

'''
#Can be G.maxGrad * Nx=readout_time
#Match the bandwidth of two sequence
BW_per_pixel=500
BW=BW_per_pixel*Nx
#Round the dwell time
dwell_time=math.ceil(1/BW/system.grad_raster_time)*system.grad_raster_time
readout_time=dwell_time*Nx
'''

gx = pp.make_trapezoid(
    channel="x", system=system, flat_area=k_width, flat_time=readout_time
)
adc = pp.make_adc(
    num_samples=Nx, system=system, duration=gx.flat_time, delay=gx.rise_time
)

# Pre-phasing gradients
pre_time = 8e-4
gz_reph = pp.make_trapezoid(
    channel="z", system=system, area=-gz.area / 2, duration=pre_time
)
# Do not need minus for in-plane prephasers because of the spin-echo (position reflection in k-space)
gx_pre = pp.make_trapezoid(
    channel="x", system=system, area=gx.area / 2 - delta_k / 2, duration=pre_time
)
gy_pre = pp.make_trapezoid(
    channel="y", system=system, area=Ny / 2 * delta_k, duration=pre_time
)

# Phase blip in shortest possible time
dur = math.ceil(2 * math.sqrt(delta_k / system.max_slew) / 10e-6) * 10e-6
gy = pp.make_trapezoid(channel="y", system=system, area=delta_k, duration=dur)

#`Refocusing pulse with spoiling gradients
rf180 = pp.make_block_pulse(
    flip_angle=np.pi, system=system, duration=500e-6, use="refocusing"
)
gz_spoil = pp.make_trapezoid(
    channel="z", system=system, area=gz.area * 2, duration=3 * pre_time
)

# Calculate delay time
duration_to_center = (Nx / 4 + 0.5) * pp.calc_duration(
    gx
) + 1 / 2 * pp.calc_duration(gy)
rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]
delay_TE1 = math.ceil((
    TE / 2
    - pp.calc_duration(gz)
    + rf_center_incl_delay
    - pre_time
    - pp.calc_duration(gz_spoil)
    - rf180_center_incl_delay
)/system.grad_raster_time)*system.grad_raster_time
delay_TE2 = math.ceil((
    TE / 2
    - pp.calc_duration(rf180)
    + rf180_center_incl_delay
    - pp.calc_duration(gz_spoil)
    - duration_to_center
)/system.grad_raster_time)*system.grad_raster_time

assert(delay_TE1>=0)
assert(delay_TE2>=0)

'''
% diffusion weithting calculation
% delayTE2 is our window for small_delta
% delayTE1+delayTE2-delayTE2 is our big delta
% we anticipate that we will use the maximum gradient amplitude, so we need
% to shorten delayTE2 by gmax/max_sr to accommodate the ramp down
'''

small_delta=delay_TE2-math.ceil(system.max_grad/system.max_slew/system.grad_raster_time)*system.grad_raster_time
big_delta=delay_TE1+pp.calc_duration(rf180,gz_spoil)
#we define bFactCalc function below to eventually calculate time-optimal
#gradients. for now we just abuse it with g=1 to give us the coefficient
#b50
g_50=np.sqrt(bFactor50*1e6/bFactCalc(1,small_delta,big_delta))
gr_50=math.ceil(g_50/system.max_slew/system.grad_raster_time)*system.grad_raster_time
#b500
g_500=np.sqrt(bFactor500*1e6/bFactCalc(1,small_delta,big_delta))
gr_500=math.ceil(g_500/system.max_slew/system.grad_raster_time)*system.grad_raster_time

gDiff_50_x=pp.make_trapezoid(channel='x',amplitude=g_50,rise_time=gr_50,flat_time=small_delta-gr_50,system=system)
gDiff_50_y=pp.make_trapezoid(channel='y',amplitude=g_50,rise_time=gr_50,flat_time=small_delta-gr_50,system=system)
gDiff_50_z=pp.make_trapezoid(channel='z',amplitude=g_50,rise_time=gr_50,flat_time=small_delta-gr_50,system=system)
gDiff_500_x=pp.make_trapezoid(channel='x',amplitude=g_500,rise_time=gr_500,flat_time=small_delta-gr_500,system=system)
gDiff_500_y=pp.make_trapezoid(channel='y',amplitude=g_500,rise_time=gr_500,flat_time=small_delta-gr_500,system=system)
gDiff_500_z=pp.make_trapezoid(channel='z',amplitude=g_500,rise_time=gr_500,flat_time=small_delta-gr_500,system=system)

duration_to_center = 0.5 * pp.calc_duration(
    gx
) 
delay_TE2 = math.ceil((
    TE / 2
    - pp.calc_duration(rf180)
    + rf180_center_incl_delay
    - pp.calc_duration(gz_spoil)
    - duration_to_center
)/system.grad_raster_time)*system.grad_raster_time

assert(pp.calc_duration(gDiff_50_x)<=delay_TE1)
assert(pp.calc_duration(gDiff_50_x)<=delay_TE2)
assert(pp.calc_duration(gDiff_500_x)<=delay_TE1)
assert(pp.calc_duration(gDiff_500_x)<=delay_TE2)

delayTR= math.ceil( (TR-pp.calc_duration(gz)  -pp.calc_duration(gx)/2 -TE +gz.fall_time +gz.flat_time/2 )/system.grad_raster_time)*system.grad_raster_time

#%%
# ======
# CONSTRUCT SEQUENCE
# ======
# Define sequence blocks
#for gDiff in [gDiff_50_x,gDiff_50_y,gDiff_50_z,gDiff_500_x,gDiff_500_y,gDiff_500_z]:
for gDiff in [gDiff_50_y,gDiff_500_y]:

    for i in range(Ny):
        seq.add_block(rf, gz)
        seq.add_block(gx_pre, gy_pre, gz_reph)
        seq.add_block(pp.make_delay(delay_TE1),gDiff)
        seq.add_block(gz_spoil)
        seq.add_block(rf180)
        seq.add_block(gz_spoil)
        seq.add_block(pp.make_delay(delay_TE2),gDiff)
        seq.add_block(gx, adc)  # Read one line of k-space
        #seq.add_block(gy)  # Phase blip
        #change the gy_pre area to one line smaller: start from positive max line
        gy_pre = pp.make_trapezoid(
    channel="y", system=system, area=((Ny / 2)-(i+1)) * delta_k, duration=pre_time
)
        #gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
        seq.add_block(pp.make_delay(delayTR))


ok, error_report = seq.check_timing()
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed! Error listing follows:")
    print(error_report)

#%%
# ======
# VISUALIZATION
# ======
if plot:
    seq.plot(time_range=(400.0,400.1))

#%%
rep=seq.test_report()
print(rep)

#%%
# =========
# WRITE .SEQ
# =========
seq.set_definition('FOV', [fov, fov, slice_thickness])
if write_seq:
    seq.write(seq_filename)

# %%