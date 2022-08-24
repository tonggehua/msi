from pypulseq.Sequence.sequence import Sequence
from math import pi
import numpy as np
from scipy.io import savemat, loadmat
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
import math
import warnings
import sigpy.mri.rf as rf_ext
import sigpy2pulseq as sp
import matplotlib.pyplot as plt


# TODO get this done today; simulate with df spingroup!
def write_PRFS(Nr, Np, FOV, thk, slice_locs, TR, TE, tau,
               enc='xyz'):
    """Proton Resonance Frequnecy Shift sequence
       For MR thermometry.

    Parameters
    ----------
    Nr : int
        readout matrix size
    Np : int
        phase encoding matrix size
    FOV : array_like
        Field-of-view in [meters];
        length-2 array with FOV[0] = FOV_readout
                            FOV[1] = FOV_pe
    slice_locs : array_like
        Slice locations in [meters]

    Returns
    -------
    seq : pypulseq Sequence
        PRFS seq object
    """

    # System reqs
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=100e-6,  # changed from 30e-6
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)
    ramp_time = 250e-6 # 0.25 ms
    dk = 1/FOV
    k_width = Nr*dk
    n_slices = len(slice_locs)

    ## Spatial encoding directions
    ch_ro = enc[0]
    ch_pe = enc[1]
    ch_ss = enc[2]

    #
    # Flip angles
    flip_ex = 90 * np.pi / 180  # degrees
    flip_ref = 180 * np.pi / 180 # degrees

    # Bloch Durations
    readout_time = 6.4e-3 + 2 * system.adc_dead_time
    t_ex = 2.5e-3
    t_ref = 2.5e-3
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time

    # RF pulses
    rf_ex_phase = np.pi/2
    rf_ref_phase = 0
    rf_ex, g_ss_ex, g_ss_ex_sr = make_sinc_pulse(flip_angle=flip_ex,system=system, duration=t_ex, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=4, phase_offset=rf_ex_phase,return_gz=True)

    rf_ref, g_ss_ref, _ = make_sinc_pulse(flip_angle=flip_ref, system=system,duration=t_ref, slice_thickness=thk,
                                          apodization=0.5, time_bw_product=4, phase_offset=rf_ref_phase,return_gz=True)

    # Calculate delays
    # Solve for t1, t2 in TE and tau
    if tau >= TE:
        raise ValueError("Tau should be less than TE")
    t1 = (TE + tau)/2
    t2 = (TE - tau)/2

    # Readout gradient and ADC
    G_ro = make_trapezoid(channel=ch_ro, system=system, flat_area = k_width, flat_time=readout_time, rise_time=ramp_time)
    adc = make_adc(num_samples=Nr, duration=G_ro.flat_time, delay=G_ro.rise_time)


    # Make phase encoding steps
    pe_steps = np.arange(1,Np+1) - 0.5*Np - 1 # from -Np/2 to +Np/2 - 1
    dk_pe = 1 / (FOV)
    pe_areas = dk_pe*pe_steps
    print(pe_areas)
    t_pe = 2e-3
    G_pe_model = make_trapezoid(channel=ch_pe, system=system, area=pe_areas[0], duration=t_pe,
                                            rise_time=ramp_time)


    # Calculate delays using component durations
    delayREF = make_delay(t1 - 0.5*t_exwd- calc_duration(G_pe_model) - 0.5*t_refwd)
    delayTE = make_delay(t2 - 0.5*t_refwd- 0.5*calc_duration(G_ro))
    delayTR = make_delay(TR - TE - 0.5*calc_duration(G_ro) - 0.5*t_exwd)

    # Add blocks
    # Single slice
    for s in range(n_slices):
        # Shift RF
        rf_ex.freq_offset = g_ss_ex.amplitude * slice_locs[s]
        rf_ref.freq_offset = g_ss_ref.amplitude * slice_locs[s]
        rf_ex.phase_offset = rf_ex_phase - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
        rf_ref.phase_offset = rf_ref_phase - 2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]

        for p in range(Np): # For each phase encoding line
            # Add 90 deg pulse
            seq.add_block(rf_ex,g_ss_ex)
            # Add Phase Encoding Gradient and delay
            G_pe = make_trapezoid(channel=ch_pe, system=system, area=pe_areas[p], duration=t_pe,
                                        rise_time=ramp_time)

            seq.add_block(g_ss_ex_sr, G_pe)
            seq.add_block(delayREF)
            seq.add_block(rf_ref,g_ss_ref)
            seq.add_block(delayTE)
            seq.add_block(G_ro, adc)
            seq.add_block(delayTR)

    return seq



if __name__ == '__main__':
    # 1995 paper parameters: GRE-based; TR = 115 ms, TE = tau = 13 ms, 128 x 128, 5 slices

    seq = write_PRFS(Nr = 15, Np = 15, FOV = 0.25,thk = 0.005, slice_locs = [0],TR = 1, TE = 0.05, tau = 0.01)
    print(seq.test_report())
    seq.write("PRFS_tau10ms.seq")
    #seq.plot(time_range=[2,2.07])