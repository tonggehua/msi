# Gehua Tong, Sept 5 2019
# Parametrized version of SPGR

from math import pi
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts


def make_pulseq_spgr_grad_spoil(fov,n,thk,fa,tr,te,enc='xyz',spoil_area='2gss',rf_spoil=False,slice_locs=[0],
                                pre_block_duration=2e-3,suffix='', write=False):
    """
    Parameters
    ----------
    fov : float
        Isotropic field-of-view (m)
    n : int
        Isotropic matrix size
    thk : float
        Slice thickness (m)
    fa : float
        Flip angle (degrees)
    tr : float
        Repetition time (s)
    te : float
        Echo time (s)
    enc : str, optional
        Orthogonal spatial encoding
        e.g. default 'xyz' means sagittal (x) readout,
                                 coronal (y) phase encoding,
                               & axial (z) slice selection.
    spoil_area : str or float
        Spoiling gradient area. If str, it must be either '2gss' or '2pi_per_voxel' or 'GE_match'
        If numerical, it has units of (1/m) since gradients in pulseq have units of Hz/m
    rf_spoil : bool
        Whether to add in RF spoiling
    slice_locs : array_like, optional
        Off-center locations of slices  (m)
        Default is one slice at the center of FOV
    write : bool
        Whether to write Sequence object into .seq file (default is False)

    Returns
    -------
    seq : Sequence
        Pypulseq Sequence object containing instructions to the scanner

    """
    SS_SPOIL_RATIO = 0.1594
    RO_SPOIL_RATIO = 0.1649

    GAMMA_BAR = 42.5775e6
    GAMMA = 2 * pi * GAMMA_BAR


    # System options for June (copied from : amri-sos service form)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                  slew_unit='T/m/s', rf_ringdown_time=30e-6,
                  rf_dead_time=100e-6, adc_dead_time=20e-6)
    seq = Sequence(system)


    ch_ro, ch_pe, ch_ss = enc[0], enc[1], enc[2]
    flip = fa * pi / 180

    # Derived sequence parameters
    Nf, Np = (n, n)
    delta_k_ro, delta_k_pe = (1 / fov, 1 / fov)
    kWidth_ro = Nf * delta_k_ro

    # # Slice select: RF and gradient
    # rf, g_ss, gssr = make_sinc_pulse(flip_angle=flip, system=system, duration=4e-3, slice_thickness=thk,
    #                                  apodization=0.5, time_bw_product=4, return_gz=True)



    rf, g_ss, gssr = make_gauss_pulse(flip_angle=flip, system=system, duration=1.164e-3, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=4, return_gz=True)
    g_ss.channel = ch_ss

    # Readout and ADC
    dwell = 10e-6
    # readoutTime = 6.4e-3
    # g_ro = make_trapezoid(channel=ch_ro, system=system, flat_area=kWidth_ro, flat_time=readoutTime)
    g_ro = make_trapezoid(channel=ch_ro, system=system, flat_area=kWidth_ro, flat_time=dwell * Nf)
    # adc = make_adc(num_samples=Nf, duration=g_ro.flat_time, delay=g_ro.rise_time)
    adc = make_adc(num_samples=Nf, dwell=dwell, delay=g_ro.rise_time)

    # Readout rewinder gradient
    pre_dur = pre_block_duration
    g_ro_pre = make_trapezoid(channel=ch_ro, system=system, area=-g_ro.area / 2, duration=pre_dur)
    # Slice refocusing gradient
    g_ss_reph = make_trapezoid(channel=ch_ss, system=system, area=-g_ss.area / 2, duration=pre_dur)
    # Phase encoding gradient areas (prep)
    phase_areas = (np.arange(Np) - (Np / 2)) * delta_k_pe
    pe_dur = pre_dur

    # print("Original area: " + str(2*pi/(fov/n)))
    # print("New area: " + str(2*g_ss.area))
    # ss_spoil_grad_area = 2*g_ss.area

    if spoil_area == '2gss':
        ss_spoil_grad_area = 2*g_ss.area
        ro_spoil_grad_area = 0

    elif spoil_area == '2pi_per_voxel':
        ss_spoil_grad_area = 1 / (fov / n)
        ro_spoil_grad_area = 0

    elif type(spoil_area) == float:
        ss_spoil_grad_area = spoil_area
        ro_spoil_grad_area = spoil_area

    elif spoil_area == 'GE_match':
        ss_spoil_grad_area = SS_SPOIL_RATIO*g_ss.area
        ro_spoil_grad_area = RO_SPOIL_RATIO*g_ss.area
    else:
        raise ValueError("spoil_area must be either 2gss or 2pi_per_voxel if it's a string, or it must be float")


    g_ss_spoil = make_trapezoid(channel=ch_ss, system=system, area=ss_spoil_grad_area, duration=pe_dur)
    g_ro_spoil = make_trapezoid(channel=ch_ro, system=system, area=ro_spoil_grad_area, duration=pe_dur)

    # Timing : delays!
    delayTE = te - calc_duration(g_ro_pre) - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2
    delayTR = tr - te - calc_duration(g_ro) / 2 - calc_duration(g_ss) / 2 - pe_dur

    delay1 = make_delay(delayTE)
    delay2 = make_delay(delayTR)
    # ro_spoil_grad_area = 0 * 2*pi/(GAMMA*fov/Nf)
    # g_ro_spoil = make_trapezoid(channel=ch_ro,system=system,area=ro_spoil_grad_area,duration=2.4e-3)

    # RF spoiling
    psi = 117 * pi / 180
    rf_spoil_phases = psi + psi * (np.arange(0, Np) * np.arange(1, Np + 1) / 2)

    # Construct sequence!
    for u in range(len(slice_locs)):  # for each slice
        # add frequency offset to RF pulse
        rf.freq_offset = g_ss.amplitude * slice_locs[u]
        for i in range(Np): # for each phase encode
            if rf_spoil:
                # add phase offset for RF spoiling
                rf.phase_offset = rf_spoil_phases[i]
                adc.phase_offset = rf_spoil_phases[i]

            # Construct corresponding phase encoding gradients (encoding and cancelling lobes)
            g_pe = make_trapezoid(channel=ch_pe, system=system, area=phase_areas[i], duration=pe_dur)
            # if not spoiling, make g_pe_neg a zero gradient
            g_pe_neg_area = 0 if spoil_area==0 else -phase_areas[i]
            g_pe_neg = make_trapezoid(channel=ch_pe, system=system, area=g_pe_neg_area,
                                      duration=pe_dur)  # Phase rewinder, for same gradient area across TRs!
            # Add events!
            seq.add_block(rf, g_ss) # RF pulse
            seq.add_block(g_pe, g_ro_pre, g_ss_reph)  # phase encoding, readout rewinder, slice rephasing
            seq.add_block(delay1)  # pre-TE delay
            seq.add_block(g_ro, adc)  # Signal readout
            seq.add_block(g_pe_neg, g_ss_spoil, g_ro_spoil) # Gradient spoiling in slice selecting axis & refocusing phase encoding
            seq.add_block(delay2)  # post TE & pre TR delay

    seq.plot(time_range=[0, 2 * tr])
    if write:
        seq.write("spgr/spgr_gspoil_N{:d}_Ns{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:d}deg_{:s}.seq".format(n, len(slice_locs), te * 1000, tr * 1000, fa, suffix))
        print("Sequence stored in spgr folder")

    print('SPGR (gradient-spoiled) sequence constructed')
    return seq

def make_pulseq_spgr_rf_spoil(fov,n,thk,fa,tr,te,enc='xyz',slice_locs=[0], include_adc_phase=False,
                                pre_block_duration=2e-3,suffix='',write=False):
    GAMMA_BAR = 42.5775e6
    GAMMA = 2 * pi * GAMMA_BAR

    ch_ro, ch_pe, ch_ss = enc[0], enc[1], enc[2]

    # System options (copied from : amri-sos service form)
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                  rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

    seq = Sequence(system)

    # Input sequence parameters
    flip = fa * pi / 180
    TE = te
    TR = tr

    # Derived sequence parameters
    Nf, Np = (n, n)
    delta_k_ro, delta_k_pe = (1 / fov, 1 / fov)
    kWidth_ro = Nf * delta_k_ro

    # # Slice select: RF and gradient
    rf, g_ss, gssr = make_sinc_pulse(flip_angle=flip, system=system, duration=4e-3, slice_thickness=thk,
                                      apodization=0.5, time_bw_product=4, return_gz=True)

    g_ss.channel = ch_ss

    # Readout and ADC
    dwell = 10e-6
    # readoutTime = 6.4e-3
    g_ro = make_trapezoid(channel=ch_ro, system=system, flat_area=kWidth_ro, flat_time=Nf * dwell)
    adc = make_adc(num_samples=Nf, duration=g_ro.flat_time, delay=g_ro.rise_time)

    # Readout rewinder gradient
    g_ro_pre = make_trapezoid(channel=ch_ro, system=system, area=-g_ro.area / 2, duration=2e-3)

    # Slice refocusing gradient
    g_ss_reph = make_trapezoid(channel=ch_ss, system=system, area=-g_ss.area / 2, duration=2e-3)

    # Phase encoding gradient areas (prep)
    phase_areas = (np.arange(Np) - (Np / 2)) * delta_k_pe

    # Timing : delays!
    delayTE = TE - calc_duration(g_ro_pre) - calc_duration(g_ss) / 2 - calc_duration(g_ro) / 2

    pe_dur = 2e-3
    delayTR = TR - calc_duration(g_ro_pre) - calc_duration(g_ss) - calc_duration(g_ro) - delayTE - pe_dur

    delay1 = make_delay(delayTE)
    delay2 = make_delay(delayTR)

    # RF spoiling
    psi = 117 * pi / 180
    rf_spoil_phases = psi + psi * (np.arange(0, Np) * np.arange(1, Np + 1) / 2)

    # Construct sequence!
    for u in range(len(slice_locs)):  # keep it multi-slice
        # add frequency offset
        rf.freq_offset = g_ss.amplitude * slice_locs[u]
        for i in range(Np):
            # add phase offset for RF spoiling
            rf.phase_offset = rf_spoil_phases[i]

            if include_adc_phase:
                adc.phase_offset = rf_spoil_phases[i]


            seq.add_block(rf, g_ss)

            g_pe = make_trapezoid(channel=ch_pe, system=system, area=phase_areas[i], duration=pe_dur)
            g_pe_neg = make_trapezoid(channel=ch_pe, system=system, area=-phase_areas[i],
                                      duration=pe_dur)  # Phase rewinder, for same gradient area across TRs!

            seq.add_block(g_pe, g_ro_pre, g_ss_reph)  #
            seq.add_block(delay1)
            seq.add_block(g_ro, adc)
            seq.add_block(g_pe_neg)

            seq.add_block(delay2)

    seq.plot(time_range=[0, 2 * TR])
    if write:
        seq.write('spgr/spgr_rfspoil_N{:d}_Ns{:d}_TE{:.0f}ms_TR{:.0f}ms_FA{:d}deg_117_{:s}.seq'.format(n, len(slice_locs), TE * 1e3,
                                                                                         TR * 1e3, fa, suffix))
        np.save(f'spgr/spgr_rf_phases_{n}.npy', rf_spoil_phases)
    print('SPGR sequence (rf spoiling) constructed')

    return seq, rf_spoil_phases


if __name__ == "__main__":


    # # 110221
    # fov = 250e-3
    # n = 32
    # fa = 15
    # tr = 50e-3
    # te = 10e-3
    # thk = 5e-3
    #
    # seq, rf_spoil_phases = make_pulseq_spgr_rf_spoil(fov, n, thk, fa, tr, te, enc='xyz', slice_locs=[0], include_adc_phase=False,
    #                           pre_block_duration=2e-3, suffix='', write=False)
    # print(seq.test_report())
    # seq.plot(time_range=[0,2*tr])
    # seq.write('spgr_sim_110221_no_adc_phase.seq')

    # July 2022, PRFS
    # seq = make_pulseq_spgr_grad_spoil(fov=0.25,n=256,thk=5e-3,
    #                                   fa=30,tr=100e-3,te=10e-3,
    #                                   enc='xyz',spoil_area='2gss',
    #                                   slice_locs=[0],
    #                                     pre_block_duration=2e-3,suffix='',write=False)
    # seq.plot(time_range=[0,100e-3])


    # Aug 2022, PRFS
    seq = make_pulseq_spgr_grad_spoil(fov=0.25,n=256,thk=5e-3,
                                      fa=30,tr=100e-3,te=5e-3,
                                      enc='xyz',spoil_area='GE_match', rf_spoil=True,
                                      slice_locs=[0],
                                        pre_block_duration=2e-3,suffix='',write=False)
    seq.plot(time_range=[0,100e-3])
