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
import pickle
from odl.discr import (Gradient, uniform_discr)
from odl.tomo import (RayTransform, fbp_op, tam_danielson_window)
from odl.operator import (BroadcastOperator, power_method_opnorm,
                          ReductionOperator)
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


path = '/home/chong/SwedenWork_Chong/Data_S/20161207_clean'

phantomName = '70100644Phantom_labelled_no_bed.nii'

# Set geometry parameters
pitch_mm = 6.6
nturns = 23
volumeSize = np.array([230.0, 230.0, 140.0])
volumeOrigin = np.array([-115.0, -115.0, 0]) 

'''
detectorOrigin = np.array([-300.0, -12.0])
pixelSize = np.array([1.2, 1.2])
sourceAxisDistance = 542.8
detectorAxisDistance = 542.8
'''

# Discretization parameters
nVoxels = np.array([230, 230, 140])
nPixels = [500, 20]

'''
nProjection = 4000 * n_turns
detectorSize = pixelSize * nPixels
'''

# Discrete reconstruction space
reco_space = uniform_discr(volumeOrigin, volumeOrigin + volumeSize,
                           nVoxels, dtype='float32', interp='linear')

# Create forward operator based on the geometry from files
turns = range(6, 15) 
proj_ops = []
for turn in turns:
    phantomStart = '/helical_proj_70100644Phantom_labelled_no_bed_Sim_num_0_Turn_num_'
    fileEnd = '_geometry.p'
    geomFile = path + phantomStart + str(turn) + fileEnd
    geom = pickle.load(open(geomFile, 'rb'), encoding='latin1') 
    proj_ops += [RayTransform(reco_space, geom, impl='astra_cuda')]          

forward_op = BroadcastOperator(*proj_ops)

#Create FBP method
FBP = ReductionOperator(*[(fbp_op(proj_op, padding=True, filter_type='Hamming',
                                  frequency_scaling=0.8) * 
                                  tam_danielson_window(proj_op))
                                  for proj_op in forward_op])

# Get data from files
imagesTurn = []
for turn in turns:   
    print(turn)
    # Forward projection
    phantomNameStart = '/helical_proj_70100644Phantom_labelled_no_bed_90_Simulations_Turn_num_'
    fileEnd = '.npy'
    imageDataFile = path + phantomNameStart + str(turn) + fileEnd
    projections = np.load(imageDataFile).astype('float32')
    projections[projections == 0] = 1e-9
    imagesTurn += [-np.log(projections / 6600)]

data = forward_op.range.element(imagesTurn)

# Initialize gradient operator
grad_op = Gradient(reco_space, method='forward')

# Column vector of two operators
op = BroadcastOperator(forward_op, grad_op)

# Do not use the g functional, set it to zero.
g = ZeroFunctional(op.domain)

# Estimate operator norm
forward_op_norm = power_method_opnorm(forward_op[1], maxiter=10)
grad_op_norm = power_method_opnorm(grad_op, maxiter=10)

# Set regularization parameter
lamb = 0.005

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = lamb * L1Norm(grad_op.range)

# --- Select solver parameters and solve using Chambolle-Pock --- #
# Estimate operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.5 * np.sqrt(len(forward_op.operators)*(forward_op_norm**2) + grad_op_norm**2)
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
    l2_norm = L2NormSquared(forward_op.range).translated(data + v)

    # Create functionals for the dual variable
    # Combine functionals, order must correspond to the operator K
    f = SeparableSum(l2_norm, l1_norm)
    
    # Optionally pass callback to the solver to display intermediate results
    callback = (CallbackPrintIteration() &
                CallbackShow(coords=[None, 0, None]) &
                CallbackShow(coords=[0, None, None]) &
                CallbackShow(coords=[None, None, 60]))

    # Run the algorithm
    chambolle_pock_solver(
        x, f, g, op, tau=tau, sigma=sigma, niter=interior_niter, gamma=gamma,
        callback=callback)
    
    # Update subgradient
    v = v - forward_op(x) + data

# Show final result
x.show(coords=[None, None, 75], title='Reconstructed result by Bregman-TV with 300 iterations')

# Get the result from FBP method
fbp_result = FBP(data)
fbp_result.show(title='Reconstructed result by FBP method')

