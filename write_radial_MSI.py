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

from write_2D_MSI import reorder_to_center_out, set_June_system_limits, calculate_MSI_rf_params
from sequence_helpers import make_oblique_gradients, modify_gradient, get_ktraj_with_rew


# Make a radial SE-based MSI sequence
# Full pulse, asymmetric spokes
# Initialize
# Define parameters
def make_2d_radial_MSI(FOV=250e-3, N=128, thk=5e-3, sl_loc=0, enc='xyz',
                       TR=500e-3, TE=7e-3, bin_width=800, N_bins=1,
                       spoiling=None, t_ex=2.5e-3, t_ref=2e-3):

    # spoiling : None, 'rf', 'grad', 'both'
    # Initialize ----------------------------------------------------------------
    system = set_June_system_limits()
    seq = Sequence(system)
    ch_ro1 = enc[0]
    ch_ro2 = enc[1]
    ch_ss = enc[2]

    ## MSI bins
    randomize_bin = False
    bin_order_center_out = True
    rfbw_factor = 1

    ## Radial k-space
    #Nr = 804
    Nr = round(np.pi*N) # Nyquist sampling
    ro_asymmetry = 0.97

    # 3. Calculate dependent parameters & define other non-custom sequence settings
    ## Basics
    dk = 1 / FOV
    #k_width = N * dk

    ## Radial k-space
    dphi = 2 * np.pi / Nr #
    ro_duration = 2.5e-3
    ro_os = 2 # Oversampling factor
    # Asymmetry! (0 - fully rewound; 1 - half-echo)
    Nro = np.round(ro_os * N)  # Number of readout points
    s = np.round(ro_asymmetry * Nro / 2) / (Nro / 2)
    dk = (1 / FOV) / (1 + s)
    ro_area = N * dk
    r = 0.5*(1-s)

    ## MSI Bins
    # Makes bin center array and fix RF bandwidth
    bin_centers = bin_width * (np.arange(N_bins) - (N_bins - 1) / 2)
    if randomize_bin:
        np.random.shuffle(bin_centers)
    if bin_order_center_out:
        print(f'Bin centers: {bin_centers}')
        bin_centers = reorder_to_center_out(bin_centers)
        print(f'Bin centers, reordered: {bin_centers}')
    rfbw = bin_width * rfbw_factor # fix RF bandwidth 

    # Others
    #ramp_time = 250e-6
    ramp_time = 100e-6
    #readout_time = 6.4e-3 + 2 * system.adc_dead_time
    rf_ex_phase = np.pi / 2


    # 4. Calculate sequence components
    ## RF
    # FROM radial
    # From msi
    ## RF 90
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_exwd = round(t_exwd /1e-5) * 1e-5
    rf90, g_ss, gss90ref = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=t_ex, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=t_ex * rfbw, phase_offset=rf_ex_phase,
                                     return_gz=True)
    gss90 = make_trapezoid(channel=ch_ss, system=system, amplitude=-g_ss.amplitude, flat_time=t_exwd,
                           rise_time=ramp_time)  # Note that the 90-deg SS gradient's amplitude is reversed for 2D MSI purposes!!
    modify_gradient(gss90ref, scale=-1, channel=ch_ss)

    # plt.figure(1)
    # plt.plot(rf90.t, np.absolute(rf90.signal))
    # plt.plot(rf90.t, np.angle(rf90.signal))
    # plt.show()

    ## RF 180
    rf_ref_phase = 0
    rf180, gz, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=t_ref, slice_thickness=thk,
                                        apodization=0.5, time_bw_product=t_ref * rfbw, phase_offset=rf_ref_phase,
                                        use='refocusing',return_gz=True)
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    gss180 = make_trapezoid(channel=ch_ss, system=system, amplitude=g_ss.amplitude, flat_time=t_refwd,
                            rise_time=ramp_time)
    # plt.figure(1)
    # plt.plot(rf180.t, np.absolute(rf180.signal))
    # plt.plot(rf180.t, np.angle(rf180.signal))
    # plt.show()

    ## Readout and pre-phaser
    gro = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
    adc = make_adc(num_samples=Nro, duration=gro.flat_time, delay=gro.rise_time, system=system)
    gro_pre = make_trapezoid(channel='x', area= (gro.area - ro_area) / 2 + (ro_area / 2) * (1 - s), system=system) # positive area because it precedes 180

    # Delays
    readout_left_dur = gro.flat_time * r/(r+1) + gro.rise_time
    readout_right_dur = gro.flat_time * 1/(r+1) + gro.rise_time

    # Spoiling
    # Grad spoil
    gss_spoil = make_trapezoid(channel=ch_ss, area=2*gss90.area)
    # RF spoil
    psi = 117 * pi / 180
    rf_spoil_phases = psi + psi * (np.arange(0, Nr) * np.arange(1, Nr + 1) / 2)

    delay180_time = TE/2 - 0.5*calc_duration(gss180) - 0.5*calc_duration(gss90) - \
                          max([calc_duration(gro_pre),calc_duration(gss90ref)])
    delay180 = make_delay(np.round(delay180_time,5))

    delayTE_time = TE/2 - 0.5*calc_duration(gss180) - readout_left_dur
    delayTE = make_delay(np.round(delayTE_time,5))

    delayTR_time = TR - TE - readout_right_dur - 0.5*calc_duration(gss90)
    delayTR = make_delay(np.round(delayTR_time,5))

    # Set up ktraj
    ktraj = np.zeros((Nr, int(adc.num_samples)),dtype=complex)


    # for each bin
    for b in range(N_bins):
        print('adding bin')
        # Set RF f0 for bin
        offset_90, offset_180 = calculate_MSI_rf_params(z0=sl_loc, dz=thk, f0=bin_centers[b], df=bin_width)
        adc.freq_offset = bin_centers[b]

        rf90.freq_offset = offset_90
        rf90.phase_offset = rf_ex_phase - 2 * np.pi * rf90.freq_offset * calc_rf_center(rf90)[0]

        rf180.freq_offset = offset_180
        rf180.phase_offset = rf_ref_phase - 2 * np.pi * rf180.freq_offset * calc_rf_center(rf180)[0]

        # For each spoke
        for spoke_ind in range(Nr):
            # Accumulate RF spoiling phase if using RF spoiling
            if spoiling in ['rf','both']:
                rf90.phase_offset = rf_ex_phase - 2 * np.pi * rf90.freq_offset * calc_rf_center(rf90)[0] + rf_spoil_phases[spoke_ind]
                rf180.phase_offset = rf_ref_phase - 2 * np.pi * rf180.freq_offset * calc_rf_center(rf180)[0] + rf_spoil_phases[spoke_ind]
                adc.phase_offset = rf_spoil_phases[spoke_ind]

            phi = dphi * spoke_ind
            # Generate radial gradients
            gp1, gp2, _ = make_oblique_gradients(gro_pre, np.array([np.cos(phi),np.sin(phi),0]))
            modify_gradient(gp1,scale=1,channel=ch_ro1)
            modify_gradient(gp2,scale=1,channel=ch_ro2)
            gr1, gr2, _ = make_oblique_gradients(gro, np.array([np.cos(phi),np.sin(phi),0]))
            modify_gradient(gp1, scale=1, channel=ch_ro1)
            modify_gradient(gp2, scale=1, channel=ch_ro2)

            # Add blocks!
            seq.add_block(rf90,gss90)
            seq.add_block(gp1, gp2, gss90ref)
            seq.add_block(delay180)
            seq.add_block(rf180,gss180)
            seq.add_block(delayTE)
            seq.add_block(gr1, gr2, adc)

            if spoiling in ['grad','both']:
                seq.add_block(delayTR, gss_spoil)
            else:
                seq.add_block(delayTR)

            if b == 0:
                modify_gradient(gp1, scale=-1)
                modify_gradient(gp2, scale=-1)
                ktraj[spoke_ind, :] = get_ktraj_with_rew(gr1, gp1, gr2, gp2, adc, display=False)


    # # Check and display sequence
    seq.plot(time_range=[0,2*TE])
    print(seq.test_report())
    #
    # # Save seq and ktraj
    # seq.write('seqs/radial/radial_msi_tr500_te7_n128_thk5_800Hz_3bins_fov250.seq')
    savemat('seqs/radial/ktraj_msi_022123.mat',{'ktraj': ktraj})
    return seq


def make_2d_radial_SE(FOV=250e-3, N=128, thk=5e-3, sl_loc=0, enc='xyz',
                       TR=500e-3, TE=7e-3,
                       spoiling=None, t_ex=2.5e-3, t_ref=2e-3, use_half_pulse=False):

    if use_half_pulse:
        cp = 1
    else:
        cp = 0.5
    C = int(use_half_pulse) + 1

    # spoiling : None, 'rf', 'grad', 'both'
    # Initialize ----------------------------------------------------------------
    system = set_June_system_limits()
    seq = Sequence(system)
    ch_ro1 = enc[0]
    ch_ro2 = enc[1]
    ch_ss = enc[2]

    ## MSI bins
    randomize_bin = False
    bin_order_center_out = True
    rfbw_factor = 1

    ## Radial k-space
    #Nr = 804
    Nr = round(np.pi*N)
    ro_asymmetry = 0.97

    # 3. Calculate dependent parameters & define other non-custom sequence settings
    ## Basics
    dk = 1 / FOV
    #k_width = N * dk

    ## Radial k-space
    dphi = 2 * np.pi / Nr #
    ro_duration = 2.5e-3
    ro_os = 2 # Oversampling factor
    # Asymmetry! (0 - fully rewound; 1 - half-echo)
    Nro = np.round(ro_os * N)  # Number of readout points
    s = np.round(ro_asymmetry * Nro / 2) / (Nro / 2)
    dk = (1 / FOV) / (1 + s)
    ro_area = N * dk
    r = 0.5*(1-s)


    # Others
    #ramp_time = 250e-6
    ramp_time = 100e-6
    #readout_time = 6.4e-3 + 2 * system.adc_dead_time
    rf_ex_phase = np.pi / 2



    # 4. Calculate sequence components
    ## RF
    # FROM radial
    # From msi


    ## RF 90
    t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
    t_exwd = round(t_exwd /1e-5) * 1e-5
    rf90, gz_90, __ = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=t_ex, slice_thickness=thk,
                                     apodization=0.5, time_bw_product=3, phase_offset=rf_ex_phase,
                                     return_gz=True, center_pos=cp)
    gss90 = make_trapezoid(channel=ch_ss, system=system, amplitude=gz_90.amplitude, flat_time=t_exwd,
                           rise_time=ramp_time)  # Note that the 90-deg SS gradient's amplitude is reversed for 2D MSI purposes!!
    gss90ref = make_trapezoid(channel=ch_ss,system=system,area=-gz_90.area/2)
    gss_ramp_reph = make_trapezoid(channel=ch_ss,area=-gss90.fall_time * gss90.amplitude/2,system=system)

    #modify_gradient(gss90ref, scale=-1, channel=ch_ss)

    # plt.figure(1)
    # plt.plot(rf90.t, np.absolute(rf90.signal))
    # plt.plot(rf90.t, np.angle(rf90.signal))
    # plt.show()

    ## RF 180
    rf_ref_phase = 0
    rf180, gz_180, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=t_ref, slice_thickness=thk,
                                        apodization=0.5, time_bw_product=3, phase_offset=rf_ref_phase,
                                        use='refocusing',return_gz=True)
    t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
    gss180 = make_trapezoid(channel=ch_ss, system=system, amplitude=gz_180.amplitude, flat_time=t_refwd,
                            rise_time=ramp_time)
    # plt.figure(1)
    # plt.plot(rf180.t, np.absolute(rf180.signal))
    # plt.plot(rf180.t, np.angle(rf180.signal))
    # plt.show()

    ## Readout and pre-phaser
    gro = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
    adc = make_adc(num_samples=Nro, duration=gro.flat_time, delay=gro.rise_time, system=system)
    gro_pre = make_trapezoid(channel='x', area= (gro.area - ro_area) / 2 + (ro_area / 2) * (1 - s), system=system) # positive area because it precedes 180

    # Delays
    readout_left_dur = gro.flat_time * r/(r+1) + gro.rise_time
    readout_right_dur = gro.flat_time * 1/(r+1) + gro.rise_time

    # Spoiling
    # Grad spoil
    gss_spoil = make_trapezoid(channel=ch_ss, area=2*gss90.area)
    # RF spoil
    psi = 117 * pi / 180
    rf_spoil_phases = psi + psi * (np.arange(0, Nr) * np.arange(1, Nr + 1) / 2)



    delay180_time = TE/2 - 0.5*calc_duration(gss180) - 0.5*calc_duration(gss90) - \
                          max([calc_duration(gro_pre),calc_duration(gss90ref)])
    delay180 = make_delay(np.round(delay180_time,5))

    delayTE_time = TE/2 - 0.5*calc_duration(gss180) - readout_left_dur
    delayTE = make_delay(np.round(delayTE_time,5))

    delayTR_time = TR - TE - readout_right_dur - 0.5*calc_duration(gss90)
    delayTR = make_delay(np.round(delayTR_time,5))

    # Set up ktraj
    ktraj = np.zeros((Nr, int(adc.num_samples)),dtype=complex)


    # for each bin
    print('adding bin')
    # Set RF f0 for bin
    #offset_90, offset_180 = calculate_MSI_rf_params(z0=sl_loc, dz=thk, f0=bin_centers[b], df=bin_width)
    #adc.freq_offset = 0

    rf90.phase_offset = rf_ex_phase
    rf180.phase_offset = rf_ref_phase

    # For half pulses, C = 2; each spoke needs to be acquired twice with opposite gradients
    for c in range(C):
        # For each spoke
        for spoke_ind in range(Nr):
            # For half pulse option, reverse slice select amplitude (always z = 0)
            if use_half_pulse:
                rf90.freq_offset = -1 * rf90.freq_offset
                modify_gradient(gss90, scale=-1)
                modify_gradient(gss90ref, scale=-1)
                modify_gradient(gss_ramp_reph, scale=-1)

            # Accumulate RF spoiling phase if using RF spoiling
            if spoiling in ['rf','both']:
                rf90.phase_offset = rf_ex_phase - 2 * np.pi * rf90.freq_offset * calc_rf_center(rf90)[0] + rf_spoil_phases[spoke_ind]
                rf180.phase_offset = rf_ref_phase - 2 * np.pi * rf180.freq_offset * calc_rf_center(rf180)[0] + rf_spoil_phases[spoke_ind]
                adc.phase_offset = rf_spoil_phases[spoke_ind]

            phi = dphi * spoke_ind
            # Generate radial gradients
            gp1, gp2, _ = make_oblique_gradients(gro_pre, np.array([np.cos(phi),np.sin(phi),0]))
            modify_gradient(gp1,scale=1,channel=ch_ro1)
            modify_gradient(gp2,scale=1,channel=ch_ro2)
            gr1, gr2, _ = make_oblique_gradients(gro, np.array([np.cos(phi),np.sin(phi),0]))
            modify_gradient(gp1, scale=1, channel=ch_ro1)
            modify_gradient(gp2, scale=1, channel=ch_ro2)

            # Add blocks!
            seq.add_block(rf90,gss90)

            if use_half_pulse:
                seq.add_block(gp1, gp2, gss_ramp_reph)
            else:
                seq.add_block(gp1, gp2, gss90ref)

            seq.add_block(delay180)
            seq.add_block(rf180,gss180)
            seq.add_block(delayTE)
            seq.add_block(gr1, gr2, adc)

            if spoiling in ['grad','both']:
                seq.add_block(delayTR, gss_spoil)
            else:
                seq.add_block(delayTR)

            modify_gradient(gp1, scale=-1)
            modify_gradient(gp2, scale=-1)
            ktraj[spoke_ind, :] = get_ktraj_with_rew(gr1, gp1, gr2, gp2, adc, display=False)


    # # Check and display sequence
    seq.plot(time_range=[0,2*TE])
    #print(seq.test_report())
    #
    # # Save seq and ktraj
    # seq.write('seqs/radial/radial_msi_tr500_te7_n128_thk5_800Hz_3bins_fov250.seq')
    savemat('seqs/radial/ktraj_radialSE_031823.mat', {'ktraj': ktraj})
    return seq


if __name__ == "__main__":
    #seq = make_2d_radial_MSI(spoiling='grad',t_ex=1e-3, t_ref=1e-3,
    #                         TR=100e-3,TE=3.7e-3, bin_width=2400)
    #seq.write('msi2d_radial_shortRF_TR100_TE3.7_gradspoil_widebin2400.seq')

    seq = make_2d_radial_SE(FOV=250e-3,N=128,spoiling='rf',t_ex=1e-3,t_ref=1e-3,TR=100-3,TE=3.8e-3,use_half_pulse=False)
    seq.check_timing()
    seq.write('se2d_radial_rfspoil_halfarea_031823.seq')
    #seq.write('se2d_radial_shortRF_TR100_TE3.8_gradspoil_fixed.seq')
    #seq.write('se2d_radial_TR1000_TE3.8_forsimN16.seq')

    #seq = make_2d_radial_MSI(N=16, spoiling='rf',t_ex=1e-3, t_ref=1e-3, TR=1000e-3, TE=3.8e-3, bin_width=2400,
 #                            N_bins=1)
#    seq.write("msi2d_radial_forsimN16_longTR_030923.seq")
