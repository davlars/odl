# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
Test for Bregman-TV method.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.discr import (Gradient, uniform_discr, uniform_partition)
from odl.tomo import Parallel2dGeometry, RayTransform
from odl.phantom import (shepp_logan, white_noise)
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
                         L2NormSquared, L1Norm, SeparableSum, 
                         chambolle_pock_solver)
standard_library.install_aliases()


def smooth_square(x):
    mask = (x[0] >= -8) & (x[0] <= 8) & (x[1] >= -8) & (x[1] <= 8)
    return (mask * x[0] + 8.)/16.


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


#phantom = space.element(smooth_square)
#phantom.show()

# Discrete reconstruction space: discretized functions on the rectangle
reco_space = uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128, 128],
    dtype='float32', interp='linear')

## Create the ground truth as the Shepp-Logan phantom
#ground_truth = shepp_logan(rec_space, modified=True)

# Create the ground truth as the given image
#I0name = './pictures/edges_and_smooth.png'
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
#ground_truth = rec_space.element(I0)
ground_truth = reco_space.element(smooth_square)
ground_truth.show('ground_truth')

# Give the number of directions
num_angles = 22
    
# Create the uniformly distributed directions
angle_partition = uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, False)])
    
# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = uniform_partition(-24, 24, 182)
    
# Create 2-D parallel projection geometry
geometry = Parallel2dGeometry(angle_partition, detector_partition)
    
# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = RayTransform(reco_space, geometry, impl='astra_cuda')

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = 0.1 * white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Initialize gradient operator
grad_op = Gradient(reco_space, method='forward')

# Column vector of two operators
op = BroadcastOperator(forward_op, grad_op)

# Do not use the g functional, set it to zero.
g = ZeroFunctional(op.domain)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.03 * L1Norm(grad_op.range)

# --- Select solver parameters and solve using Chambolle-Pock --- #
# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.3 * power_method_opnorm(op, maxiter=6)

niter = 100  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
gamma = 0.2

# Choose a starting point
x = forward_op.domain.zero()

# Maximum exterior iteration number
exterior_niter = 3

# Maximum interior iteration number
interior_niter = 100

# Create subgradient at zero starting point
v = forward_op.range.zero()

# Begin exterior iterations
for _ in range(exterior_niter):
    # l2-squared data matching
    l2_norm = L2NormSquared(forward_op.range).translated(noise_proj_data + v)

    # Create functionals for the dual variable
    # Combine functionals, order must correspond to the operator K
    f = SeparableSum(l2_norm, l1_norm)
    
    # Optionally pass callback to the solver to display intermediate results
    callback = (CallbackPrintIteration() & CallbackShow())

    # Run the algorithm
    chambolle_pock_solver(
        x, f, g, op, tau=tau, sigma=sigma, niter=interior_niter, gamma=gamma,
        callback=callback)
    
    # Update subgradient
    v = v - forward_op(x) + noise_proj_data

# Show final result
x.show(title='Reconstructed result by Bregman-TV with 300 iterations')
