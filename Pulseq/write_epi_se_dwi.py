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
plot=True
write_seq=True
seq_filename = "epi_se_pypulseq.seq"

# ======
# SETUP
# ======
seq = pp.Sequence()  # Create a new sequence object
fov = 256e-3  # Define FOV and resolution
Nx = 64
Ny = 64
TE = 60e-3
bFactor50=50
bFactor500=500
slice_thickness=8e-3
Nslices=1


# Set system limits
system = pp.Opts(
    max_grad=32,
    grad_unit="mT/m",
    max_slew=130,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=20e-6,
)

# ======
# CREATE EVENTS
# ======
# Create 90 degree slice selection pulse and gradient
rf, gz, _ = pp.make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=3e-3,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    return_gz=True,
)

# Define other gradients and ADC events
delta_k = 1 / fov
k_width = Nx * delta_k
readout_time = 3.2e-4
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

# Refocusing pulse with spoiling gradients
rf180 = pp.make_block_pulse(
    flip_angle=np.pi, system=system, duration=500e-6, use="refocusing"
)
gz_spoil = pp.make_trapezoid(
    channel="z", system=system, area=gz.area * 2, duration=3 * pre_time
)

# Calculate delay time

duration_to_center = (Nx / 2 + 0.5) * pp.calc_duration(
    gx
) + Ny / 2 * pp.calc_duration(gy)
rf_center_incl_delay = rf.delay + pp.calc_rf_center(rf)[0]
rf180_center_incl_delay = rf180.delay + pp.calc_rf_center(rf180)[0]
delay_TE1 = (
        math.ceil(
            (
                TE / 2
                - pp.calc_duration(rf, gz)
                + rf_center_incl_delay
                - rf180_center_incl_delay
            )
            / system.grad_raster_time
        )
        * system.grad_raster_time
    )
delay_TE2 = (
    math.ceil(
        (
            TE / 2
            - pp.calc_duration(rf180, gz_spoil)
            + rf180_center_incl_delay
            - duration_to_center
        )
        / system.grad_raster_time
    )
    * system.grad_raster_time
)
assert delay_TE1 >= 0
assert gx_pre.delay >= pp.calc_duration(rf180)
assert pp.calc_duration(gy_pre) <= pp.calc_duration(gx_pre)

# CONSTRUCT SEQUENCE
# ======
# Define sequence blocks
seq.add_block(rf, gz)
seq.add_block(gx_pre, gy_pre, gz_reph)
seq.add_block(pp.make_delay(delay_TE1))
seq.add_block(gz_spoil)
seq.add_block(rf180)
seq.add_block(gz_spoil)
seq.add_block(pp.make_delay(delay_TE2))
for i in range(Ny):
    seq.add_block(gx, adc)  # Read one line of k-space
    seq.add_block(gy)  # Phase blip
    gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient
seq.add_block(pp.make_delay(1e-4))

ok, error_report = seq.check_timing()
if ok:
    print("Timing check passed successfully")
else:
    print("Timing check failed! Error listing follows:")
    print(error_report)

# ======
# VISUALIZATION
# ======
if plot:
    seq.plot()

# =========
# WRITE .SEQ
# =========
if write_seq:
    seq.write(seq_filename)
