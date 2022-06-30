import math

import numpy as np
import pypulseq as pp
import sigpy.mri.rf as rf_ext

import sigpy2pulseq as sp

if (ext_pulse_library):
    # Declare pulse configuration inputs here
    pulse_freq_offset = 0.0  # if designing rf pulse without a FM
    if ext_pulse_type == 'slr':
        pulse = rf_ext.slr.dzrf(n=int(round(duration / system.rf_raster_time)), tb=tb, ptype='st', ftype='ls', d1=0.01,
                                d2=0.01,
                                cancel_alpha_phs=True)

    if ext_pulse_type == 'sms':
        pulse_in = rf_ext.slr.dzrf(n=int(round(duration / system.rf_raster_time)), tb=tb, ptype='st', ftype='ls',
                                   d1=0.01, d2=0.01,
                                   cancel_alpha_phs=False)
        pulse = rf_ext.multiband.mb_rf(pulse_in, n_bands=3, band_sep=25, phs_0_pt='None')

    if ext_pulse_type == 'hypsec':
        # Let us design two pulses here
        # - STA for excitation
        pulse = rf_ext.slr.dzrf(n=int(round(duration / system.rf_raster_time)), tb=tb, ptype='st', ftype='ls', d1=0.01,
                                d2=0.01,
                                cancel_alpha_phs=True)

        # - Hyperbolic secant pulse for inversion as part of magnetization prep
        duration_hypsec = 2 * duration  # adiabatic pulses need more time
        pulse_hypsec, pulse_freq_offset = rf_ext.adiabatic.hypsec(n=2002,
                                                                  beta=500,
                                                                  mu=60, dur=duration_hypsec)

        rf_inv = sp.sig_2_seq(pulse=pulse_hypsec, flip_angle=alpha * math.pi / 180, system=system,
                              duration=duration_hypsec,
                              slice_thickness=slice_thickness,
                              return_gz=False, time_bw_product=tb, rf_freq=pulse_freq_offset)

    # Display if needed
    if (disp_pulse):
        if (ext_pulse_type == 'hypsec'):
            sp.disp_pulse(pulse_hypsec, pulse_freq_offset, tb, duration_hypsec, system)
        else:
            sp.disp_pulse(pulse, 0, tb, duration, system)

    # Convert and integrate with pypulseq
    rf, gz, gzr, _ = sp.sig_2_seq(pulse=pulse, flip_angle=alpha * math.pi / 180, system=system, duration=duration,
                                  slice_thickness=slice_thickness,
                                  return_gz=True, time_bw_product=tb, rf_freq=0)
else: