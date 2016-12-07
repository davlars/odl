#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:58:10 2016

@author: hkohr, jwbuurlage, davlars
"""

import numpy as np
import odl
from odl.util import writable_array
from builtins import super

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

from odl.discr.multires import MaskingOperator, show_both, show_extent


# %%

# Basic discretizations
min_pt = [-10, -10]
max_pt = [10, 10]
coarse_discr = odl.uniform_discr(min_pt, max_pt, [50, 50])
fine_discr = odl.uniform_discr(min_pt, max_pt, [1000, 1000])

insert_min_pt = [-2, -8]
insert_max_pt = [2, -4]

# Geometry
angle_partition = odl.uniform_partition(0, np.pi, 180)
det_partition = odl.uniform_partition(-15, 15, 3000)

geometry = odl.tomo.Parallel2dGeometry(angle_partition, det_partition,
                                       det_init_pos=[20, 0])

# Mask
coarse_mask = MaskingOperator(coarse_discr, insert_min_pt, insert_max_pt)
coarse_ray_trafo = odl.tomo.RayTransform(coarse_discr, geometry,
                                         impl='astra_cpu')
masked_coarse_ray_trafo = coarse_ray_trafo * coarse_mask

# Phantom
phantom_c = odl.phantom.shepp_logan(coarse_discr, modified=True)
phantom_f = odl.phantom.shepp_logan(fine_discr, modified=True)


# Define insert discretization using the fine cell sizes but the insert
# min and max points
insert_discr = odl.uniform_discr_fromdiscr(
    fine_discr, min_pt=insert_min_pt, max_pt=insert_max_pt,
    cell_sides=fine_discr.cell_sides)

# Restrict the phantom to the insert discr
resizing_operator = odl.ResizingOperator(fine_discr, insert_discr)
phantom_insert = resizing_operator(phantom_f)

# Ray trafo on the insert discretization only
insert_ray_trafo = odl.tomo.RayTransform(insert_discr, geometry,
                                         impl='astra_cpu')

# Forward operator = sum of masked coarse ray trafo and insert ray trafo
sum_ray_trafo = odl.ReductionOperator(masked_coarse_ray_trafo,
                                      insert_ray_trafo)

# Make phantom in the product space
pspace = sum_ray_trafo.domain
phantom = pspace.element([phantom_c, phantom_insert])

# Create noise-free data
fine_ray_trafo = odl.tomo.RayTransform(fine_discr, geometry,
                                       impl='astra_cpu')
data = fine_ray_trafo(phantom_f)
data.show('data')

# Make noisy data
noisy_data = data + odl.phantom.white_noise(fine_ray_trafo.range, stddev=0.1)
noisy_data.show('noisy data')

reco = sum_ray_trafo.domain.zero()

# %% Reconstruction
reco_method = 'CG'
if reco_method == 'CG':
    callback = odl.solvers.CallbackShow(display_step=1)
    odl.solvers.conjugate_gradient_normal(sum_ray_trafo, reco, noisy_data,
                                          niter=5, callback=callback)
    show_both(reco[0], reco[1])

elif reco_method == 'TV':
    fine_grad = odl.Gradient(insert_discr, pad_mode='order1')

    # Differentiable part, build as ||. - g||^2 o P
    data_func = odl.solvers.L2NormSquared(
        sum_ray_trafo.range).translated(noisy_data) * sum_ray_trafo
    reg_param_1 = 7e-3
    reg_func_1 = reg_param_1 * (odl.solvers.L2NormSquared(coarse_discr) *
                                odl.ComponentProjection(pspace, 0))
    smooth_func = data_func + reg_func_1

    # Non-differentiable part composed with linear operators
    reg_param = 7e-4
    nonsmooth_func = reg_param * odl.solvers.L1Norm(fine_grad.range)

    # Assemble into lists (arbitrary number can be given)
    comp_proj_1 = odl.ComponentProjection(pspace, 1)
    lin_ops = [fine_grad * comp_proj_1]
    nonsmooth_funcs = [nonsmooth_func]

    # TODO: add to nonsmooth_funcs
    box_constr = odl.solvers.IndicatorBox(insert_discr,
                                          np.min(phantom_insert),
                                          np.max(phantom_insert))
    f = odl.solvers.ZeroFunctional(pspace)

    # eta^-1 is the Lipschitz constant of the smooth functional gradient
    ray_trafo_norm = 1.1 * odl.power_method_opnorm(sum_ray_trafo,
                                                   xstart=phantom, maxiter=2)
    print('norm of the ray transform: {}'.format(ray_trafo_norm))
    eta = 1 / (2 * ray_trafo_norm ** 2 + 2 * reg_param_1)
    print('eta = {}'.format(eta))
    grad_norm = 1.1 * odl.power_method_opnorm(fine_grad, xstart=phantom_insert,
                                              maxiter=4)
    print('norm of the gradient: {}'.format(grad_norm))

    # tau and sigma are like step sizes
    sigma = 5e-3
    tau = 2.1 * sigma
    # Here we check the convergence criterion for the forward-backward solver
    # 1. This is required such that the square root is well-defined
    print('tau * sigma * grad_norm ** 2 = {}, should be <= 1'
          ''.format(tau * sigma * grad_norm ** 2))
    assert tau * sigma * grad_norm ** 2 <= 1
    # 2. This is the actual convergence criterion
    check_value = (2 * eta * min(1 / tau, 1 / sigma) *
                   np.sqrt(1 - tau * sigma * grad_norm ** 2))
    print('check_value = {}, must be > 1 for convergence'.format(check_value))
    convergence_criterion = check_value > 1
    assert convergence_criterion
    # %%
    callback = odl.solvers.CallbackShow(display_step=2)
    reco = pspace.zero()  # starting point
    odl.solvers.forward_backward_pd(reco, f=f, g=nonsmooth_funcs, L=lin_ops,
                                    h=smooth_func,
                                    tau=tau, sigma=[sigma], niter=60,
                                    callback=callback)

    show_both(reco[0], reco[1])
