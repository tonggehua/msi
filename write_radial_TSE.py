# Make a radial TSE sequence to image DBS leads with!
# Try using high-level modules? Incorporate ideas ......

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


# TODO ktraj design functions
# TODO gradient implementation
def make_2d_radial_tse(N, FOV, enc, TR, turbo_factor):
    system = set_June_system_limits()
    seq = Sequence(system)
    # Calculate derived parameters


    #

    # For each set of spokes
    for t in range(N_train):
        # Excitation
        seq.add_block()
        # Go outward

        # For each spoke
        for s in range(N_spokes[t]):
            # 180
            seq.add_block()

            # Go 'round in k-space

            # Readout

            # Go back 'round

            # Save to trajectory
            # ktraj[:,r] = ...

        # Gradient spoiling and TR delay

    return seq