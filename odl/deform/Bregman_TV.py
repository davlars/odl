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
from odl.discr import (Gradient, uniform_discr, uniform_partition)
from odl.tomo import Parallel2dGeometry, RayTransform
from odl.phantom import (shepp_logan, white_noise)
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, 
                         proximal_const_func, proximal_cconj_l2_squared,
                         combine_proximals, proximal_cconj_l1,
                         chambolle_pock_solver)
standard_library.install_aliases()


# Discrete reconstruction space: discretized functions on the rectangle
rec_space = uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')

# Create the ground truth as the Shepp-Logan phantom
ground_truth = shepp_logan(rec_space, modified=True)

# Give the number of directions
num_angles = 22
    
# Create the uniformly distributed directions
angle_partition = uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, False)])
    
# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = uniform_partition(-24, 24, 362)
    
# Create 2-D parallel projection geometry
geometry = Parallel2dGeometry(angle_partition, detector_partition)
    
# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = RayTransform(rec_space, geometry, impl='astra_cuda')

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = 0.1 * white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# TV reconstruction by Chambolle-Pock algorithm
# Initialize gradient operator
grad_op = Gradient(rec_space, method='forward')

# Column vector of two operators
op = BroadcastOperator(forward_op, grad_op)

# --- Select solver parameters and solve using Chambolle-Pock --- #
# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * power_method_opnorm(op)

tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable
gamma = 0.2

# Create the proximal operator for unconstrained primal variable
proximal_primal = proximal_const_func(op.domain)

# Isotropic TV-regularization i.e. the l1-norm
prox_convconj_l1 = proximal_cconj_l1(grad_op.range, lam=0.03, isotropic=True)

# Maximum exterior iteration number
exterior_niter = 3

# Maximum interior iteration number
interior_niter = 100

# Choose a starting point
x = forward_op.domain.zero()

# Create subgradient at zero starting point
v = forward_op.range.zero()

# Begin exterior iterations
for _ in range(exterior_niter):
    # Create proximal operators for the dual variable
    # l2-data matching
    prox_convconj_l2 = proximal_cconj_l2_squared(forward_op.range,
                                                 g=noise_proj_data + v)
    
    # Combine proximal operators, order must correspond to the operator K
    proximal_dual = combine_proximals(prox_convconj_l2, prox_convconj_l1)
    
    # Optionally pass callback to the solver to display intermediate results
    callback = (CallbackPrintIteration() & CallbackShow())

    # Begin interior iterations
    chambolle_pock_solver(
        op, x, tau=tau, sigma=sigma, proximal_primal=proximal_primal,
        proximal_dual=proximal_dual, niter=interior_niter, callback=callback,
        gamma=gamma)
    
    # Update subgradient
    v = v - forward_op(x) + noise_proj_data

# Show final result
x.show(title='Reconstructed result by Bregman-TV with 300 iterations')