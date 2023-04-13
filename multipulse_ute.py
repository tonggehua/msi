# 2D ultra-short TE sequence (radial encoding)
import numpy as np
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.opts import Opts
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_trap_pulse import make_trapezoid
from scipy.io import savemat
import copy
import matplotlib.pyplot as plt


SCANNER = 'Prisma-Skyra'

def write_multipulse_2D_UTE(N=250, Nr=128, FOV=250e-3, thk=3e-3, slice_locs=[0], FA=10, TR=10e-3, ro_asymmetry=0.97,
                            enc='xyz', TE_use=None, save_seq=True,
                            num_pulse=2, rf_factor=1, bw_pulse=200):
    """
    Single-slice 2D partially rewound UTE with RF spoiling
    With multipulse module for temperature-dependent magnitude encoding

    Parameters
    ----------
    N : int, default=250
        Matrix size
    Nr : int, default=128
        Number of radial spokes
    FOV : float, default=250
        Field-of-view in [meters]
    thk : float, default=3
        Slice thickness in [meters]
    slice_locs : float, default=[0]
        Slice location in [meters]
    FA : float, default=10
        Flip angle in [degrees]
    TR : float, default=0.01
        Repetition time in [seconds]
    ro_asymmetry : float, default=0.97
        The ratio A/B where a A/(A+B) portion of 2*Kmax is omitted and B/(A+B) is acquired.
    enc : str
        Orthogonal encoding string. 'xyz' means readout in x, phase enc. in y, and slice in z.
        Allowed: 'xyz', 'xzy', 'yzx', 'yxz', 'zxy', 'zyx'
    TE_use : float, default=None
        Desired echo time in [seconds]. If shorter than feasible, minimum TE is used.
    save_seq : bool, default=True
        Whether to save this sequence as a .seq file
    num_pulse : int
        Number of RF pulses to play at the beginning
    rf_factor : float
        Bandwidth factor for the RF pulses
    bw_pulse : float
        Nominal pulse bandwidth

    Returns
    -------
    seq : Sequence
        PyPulseq 2D UTE sequence object
    TE : float
        Echo time of generated sequence in [seconds]
    ktraj : np.ndarray
        Complex k-space trajectory (kx + 1j*ky)
        with size [spoke, readout sample]
    """

    # Set system limits
    if SCANNER == 'Prisma-Skyra':
        # Adapted from pypulseq demo write_ute.py (obtained mid-2021)
        print('Making sequence for Prisma or Skyra')
        system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
                      rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)
    elif SCANNER == 'Sola':
        print('Making sequence for Sola')
        system = Opts(max_grad=33, grad_unit='mT/m', max_slew=125, slew_unit='T/m/s',
                      rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

    seq = Sequence(system=system)

    # Derived parameters
    dphi = 2 * np.pi / Nr  #
    ro_duration = 2.5e-3
    ro_os = 2  # Oversampling factor
    rf_spoiling_inc = 117  # RF spoiling increment value.

    # Encoding
    ch_ro1 = enc[0]
    ch_ro2 = enc[1]
    ch_ss = enc[2]

    # Sequence components
    cp = 0.5

    # RF pulse, slice selecting gradient, and gradient to rephase down-ramp (half pulse only)
    # rf, gz, gz_reph = make_sinc_pulse(flip_angle=FA*np.pi/180, duration=rf_dur, slice_thickness=thk, apodization=0.5,
    #                                   time_bw_product=4, center_pos=cp, system=system, return_gz=True)
    # Make a single pulse
    tb = 2
    rf_dur = system.rf_raster_time * np.round((tb / (bw_pulse*rf_factor))/system.rf_raster_time)
    rf, gz, gz_reph = make_gauss_pulse(flip_angle=FA * np.pi / 180, duration=rf_dur, slice_thickness=thk,
                                       apodization=0.1,
                                       time_bw_product=tb, system=system, return_gz=True)
    # Cut gz into three parts, so the flat part can be replicated
    rf.delay = 0
    gz.flat_time = calc_duration(rf)
    gz.rise_time = 3e-5
    gz.fall_time = 3e-5
    gz_rise = make_extended_trapezoid(channel=ch_ss, times=[0, gz.rise_time], amplitudes=[0, gz.amplitude])
    gz_flat = make_extended_trapezoid(channel=ch_ss, times=[0, gz.flat_time], amplitudes=[gz.amplitude, gz.amplitude])
    gz_fall = make_extended_trapezoid(channel=ch_ss, times=[0, gz.fall_time], amplitudes=[gz.amplitude, 0])
    gz_reph_all = make_trapezoid(channel=ch_ss,
                                 area=-0.5 * gz.amplitude * (num_pulse * gz.flat_time + 0.5 * gz.fall_time),
                                 system=system)

    # Make sure slice selection gradients are in the correct direction
    modify_gradient(gz, scale=1, channel=ch_ss)
    # modify_gradient(gz_reph,scale=1,channel=ch_ss)

    # Calculate MSI offsets
    msi_offsets = bw_pulse * np.arange(num_pulse) - bw_pulse * (num_pulse - 1) * 0.5
    print(f'MSI offsets: {msi_offsets}')

    # Asymmetry! (0 - fully rewound; 1 - hall-echo)
    Nro = np.round(ro_os * N)  # Number of readout points
    s = np.round(ro_asymmetry * Nro / 2) / (Nro / 2)
    dk = (1 / FOV) / (1 + s)
    ro_area = N * dk
    gro = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
    adc = make_adc(num_samples=Nro, duration=gro.flat_time, delay=gro.rise_time, system=system)
    gro_pre = make_trapezoid(channel='x', area=- (gro.area - ro_area) / 2 - (ro_area / 2) * (1 - s), system=system)

    # Spoilers
    gro_spoil = make_trapezoid(channel='x', area=0.2 * N * dk, system=system)

    # Calculate timing
    # TR delay (multislice)

    TE = 0.5 * gz.flat_time + gz.fall_time + calc_duration(gro_pre,
                                                           gz_reph_all) + gro.rise_time + adc.dwell * Nro / 2 * (1 - s)

    if calc_duration(gz_reph_all) > calc_duration(gro_pre):
        gro_pre.delay = calc_duration(gz_reph_all) - calc_duration(gro_pre)
    time_per_slice = calc_duration(gz_rise) + num_pulse * calc_duration(gz_flat) + calc_duration(
        gz_fall) + calc_duration(gro_pre, gz_reph_all) + calc_duration(gro)
    delay_TR_per_slice = (TR - len(slice_locs) * time_per_slice) / len(slice_locs)

    if delay_TR_per_slice < 0:
        raise ValueError(f"TR = {TR * 1e3}ms is not long enough to accomondate {len(slice_locs)} slices!")

    delay_TR_per_slice = np.ceil(delay_TR_per_slice / seq.grad_raster_time) * seq.grad_raster_time

    # The TR delay starts at the same time as the spoilers!
    assert np.all(delay_TR_per_slice >= calc_duration(gro_spoil))

    # TE delay (if longer than minimal TE is desired)
    TE_delay = 0
    if TE_use is None:
        print(f'TE = {TE * 1e6:.0f} us')
    elif TE_use <= TE:
        print(f'Desired TE is lower than minimal TE. Using minimal TE ={TE * 1e6:.0f} us')
    else:
        TE_delay = np.ceil((TE_use - TE) / seq.grad_raster_time) * seq.grad_raster_time
        print(f'Minimal TE: {TE * 1e6:.0f} us; using TE: {(TE + TE_delay) * 1e6:.0f} us')
        TE += TE_delay

    # Starting RF phase and increments for RF spoiling
    rf_phase = 0
    rf_inc = 0

    # Set up k-space trajectory storage
    ktraj = np.zeros((Nr, int(adc.num_samples)), dtype=complex)
    u = 0

    ind_ro1 = 'xyz'.find(ch_ro1)
    ind_ro2 = 'xyz'.find(ch_ro2)

    for spoke_ind in range(Nr):  # for each spoke
        phi = dphi * spoke_ind
        # ug2d = [np.cos(phi), np.sin(phi), 0]

        ug2d = [0, 0, 0]
        ug2d[ind_ro1] = np.cos(phi)
        ug2d[ind_ro2] = np.sin(phi)
        gpx, gpy, gpz = make_oblique_gradients(gro_pre, ug2d)
        grx, gry, grz = make_oblique_gradients(gro, ug2d)
        gsx, gsy, gsz = make_oblique_gradients(gro_spoil, ug2d)
        gp = [gpx, gpy, gpz]
        gr = [grx, gry, grz]
        gs = [gsx, gsy, gsz]
        # Extract correct gradients
        gpr1 = gp[ind_ro1]
        gpr2 = gp[ind_ro2]
        grr1 = gr[ind_ro1]
        grr2 = gr[ind_ro2]
        gsr1 = gr[ind_ro1]
        gsr2 = gr[ind_ro2]

        for s in range(len(slice_locs)):  # interleaved slices
            # RF spoiling phase calculations
            rf.phase_offset = (rf_phase / 180) * np.pi
            adc.phase_offset = (rf_phase / 180) * np.pi
            # (no need to change RF frequency?)
            rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
            rf_phase = np.mod(rf_phase + rf_inc, 360.0)
            # RF slice selection freq/phase offsets
            rf.freq_offset = gz.amplitude * slice_locs[s]
            rf.phase_offset = rf.phase_offset - 2 * np.pi * rf.freq_offset * calc_rf_center(rf)[0]

            # Add blocks to seq
            seq.add_block(gz_rise)
            for p in range(num_pulse):
                # Modulate RF frequency and add
                rf.freq_offset = gz.amplitude * slice_locs[s] + msi_offsets[p]
                seq.add_block(rf, gz_flat)
            seq.add_block(gz_fall)

            if TE_use is not None:
                seq.add_block(make_delay(TE_delay))

            seq.add_block(gpr1, gpr2, gz_reph_all)
            seq.add_block(grr1, grr2, adc)
            seq.add_block(gsr1, gsr2, make_delay(delay_TR_per_slice))

        ktraj[u, :] = get_ktraj_with_rew(grr1, gpr1, grr2, gpr2, adc, display=False)

        u += 1

    ok, error_report = seq.check_timing()  # Check whether the timing of the sequence is correct
    if ok:
        print('Timing check passed successfully')
    else:
        print('Timing check failed. Error listing follows:')
        [print(e) for e in error_report]

    if save_seq:
        seq.write(f'multipulse_UTE_{num_pulse}pulses_bw{bw_pulse}_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}.seq')
        savemat(f'ktraj_multipulse_UTE_s{s}_N{N}_FOV{FOV}_TR{TR}_TE{TE}.mat',
                {'ktraj': ktraj})

    return seq, TE, ktraj



def modify_gradient(gradient,scale,channel=None):
    """Helper function to modify the strength and channel of an existing gradient

    Parameters
    ----------
    gradient : Gradient
        Pulseq gradient object to be modified
    scale : float
        Scalar to multiply the gradient strength by
    channel : str, optional {None, 'x','y','z'}
        Channel to switch gradient into; default is None
        which keeps the original channel

    """
    if gradient.type == 'trap':
        gradient.amplitude *= scale
        gradient.area *= scale
        gradient.flat_area *= scale
    elif gradient.type == 'grad':
        gradient.waveform *= scale
        gradient.first *= scale
        gradient.last *= scale

    if channel != None:
        gradient.channel = channel

def make_oblique_gradients(gradient,unit_grad):
    """
    Helper function to make oblique gradients
    (Gx, Gy, Gz) are generated from a single orthogonal gradient
    and a direction indicated by unit vector

    Parameters
    ----------
    gradient : Gradient
        Pulseq gradient object
    unit_grad: array_like
        Length-3 unit vector indicating direction of resulting oblique gradient

    Returns
    -------
    ngx, ngy, ngz : Gradient
        Oblique gradients in x, y, and z directions

    """
    ngx = copy.deepcopy(gradient)
    ngy = copy.deepcopy(gradient)
    ngz = copy.deepcopy(gradient)

    unit_grad = unit_grad / np.linalg.norm(unit_grad)

    modify_gradient(ngx, unit_grad[0],'x')
    modify_gradient(ngy, unit_grad[1],'y')
    modify_gradient(ngz, unit_grad[2],'z')

    return ngx, ngy, ngz

def get_ktraj_with_rew(gx, gx_rew, gy, gy_rew, adc, display=False):
    """
    Calculate and return one line of k-space trajectory
    during a 2D readout, accounting for rewinder gradients
    applied before readout

    Parameters
    ----------
    gx : SimpleNamespace
        Readout gradient (PyPulseq event) in the x direction
    gx_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the x direction
    gy : SimpleNamespace
        Readout gradient (PyPulseq event) in the y direction
    gy_rew : SimpleNamespace
        Rewinder gradient (PyPulseq event) in the y direction
    adc : SimpleNamespace
        ADC sampling (Pypulseq event) during readout
    display : bool, default=False
        Whether to display the trajectory in a plot

    Returns
    -------
    ktraj_complex : numpy.ndarray
        Complex representation (kx + 1j*ky) of k-space trajectory for this single readout

    """
    sampled_times = np.linspace(0, gx.flat_time, adc.num_samples, endpoint=False)
    kx_pre = 0.5 * gx.amplitude * gx.rise_time + gx_rew.area
    ky_pre = 0.5 * gy.amplitude * gy.rise_time + gy_rew.area

    kx = kx_pre + gx.amplitude * sampled_times
    ky = ky_pre + gy.amplitude * sampled_times

    if display:
        plt.figure(1001)
        plt.plot(kx, ky, 'ro-')
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.show()

    return (kx + 1j * ky)


if  __name__ == "__main__":
    enc = 'xyz'  # axial
    sl = [0]
    # Debug
    for u in range(len(sl)):
        seq, TE, ktraj = write_multipulse_2D_UTE(N=256, Nr=804, FOV=253e-3,
                                                 thk=5e-3, slice_locs=[sl[u]], FA=10, TR=50e-3,
                                                 ro_asymmetry=0.97,
                                                 enc=enc, TE_use=None, save_seq=False,
                                                 num_pulse=3, rf_factor=1.5, bw_pulse=200)

        # seq.write(f'ute2D_axial_half_at{sl[u]*1e3}mm_TE{TE*1e3}ms.seq')