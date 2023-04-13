from pypulseq.Sequence.sequence import Sequence
from pypulseq.make_delay import make_delay
from pypulseq.make_adc import make_adc
from pypulseq.make_trap_pulse import make_trapezoid
import numpy as np
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.opts import Opts

def write_slice_profile_probe(rf,gss,gssref,slice_fov,N):
    system = set_June_system_limits()
    seq = Sequence(system)
    ramp_time = 250e-6
    readout_time = 10e-3
    dk = 1/slice_fov
    k_width = N*dk
    # Excitation
    seq.add_block(rf,gss)
    if gssref is not None:
        seq.add_block(gssref)
    # Small delay
    seq.add_block(make_delay(50e-6))
    # Readout gradient and ADC
    gro = make_trapezoid(channel='z', system=system, flat_area=k_width, flat_time=readout_time,
                            rise_time=ramp_time)
    # No need for rise time delay since it is set at beginning of flat time; delay is ADC dead time
    adc = make_adc(num_samples=N, duration=gro.flat_time, delay=ramp_time)
    # Readout in slice direction
    seq.add_block(gro, adc)
    seq.add_block(make_delay(5e-2))

    return seq




def set_June_system_limits():
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                      slew_unit='T/m/s', rf_ringdown_time=100e-6, # changed from 30e-6
                      rf_dead_time=100e-6, adc_dead_time=20e-6)
    return system
if __name__ == "__main__":
    # msi_90
    system = set_June_system_limits()
    # rf, gz, gzref = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=0.001, slice_thickness=0.005,
    #                                freq_offset=0, phase_offset=0.0, time_bw_product=3, return_gz=True)
    # rf_name = 'msi_90'
    # msi_180
    #
    rf, gz, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=0.001, slice_thickness=0.005,
                                        apodization=0.5, time_bw_product=3, phase_offset=0,
                                        use='refocusing',return_gz=True)
    rf_name = 'msi_180'
    # # se_90
    # rf, gz, __ = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=2e-3, slice_thickness=5e-3,
    #                                  apodization=0.5, time_bw_product=3, phase_offset=np.pi/2,
    #                                  return_gz=True, center_pos=0.5)
    # rf_name = 'se90_2ms'

    # # se_180
    # rf, gz, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=1e-3, slice_thickness=5e-3,
    #                                    apodization=0.5, time_bw_product=3, phase_offset=0,
    #                                    use='refocusing', return_gz=True)
    # rf_name = 'se180_1ms'
    # Convert into a slice profile test!
    seq = write_slice_profile_probe(rf,gz,None,slice_fov=500e-3,N=512)

    seq.write(f'rf_probe_{rf_name}_largeFOV_040423.seq')