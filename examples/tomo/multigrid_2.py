#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:58:10 2016

@author: hkohr
"""

import numpy as np
import odl
from odl.util import writable_array
from builtins import super

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox


# %%

def show_both(coarse_data, fine_data):
    fig, ax = plt.subplots()
    
    low = np.min([np.min(coarse_data), np.min(fine_data)])
    high = np.max([np.max(coarse_data), np.max(fine_data)])
   
    normalization = mpl.colors.Normalize(vmin=low, vmax=high)
    
    ax.set_xlim(coarse_data.space.domain.min_pt[0],
                coarse_data.space.domain.max_pt[0])
    ax.set_ylim(coarse_data.space.domain.min_pt[1],
                coarse_data.space.domain.max_pt[1])

    def show(data):    
        bbox0 = Bbox.from_bounds(*data.space.domain.min_pt,
                                 *(data.space.domain.max_pt -
                                   data.space.domain.min_pt))
        bbox = TransformedBbox(bbox0, ax.transData)
        bbox_image = BboxImage(bbox, norm=normalization, cmap='bone',
                               interpolation='nearest', origin=False)
        bbox_image.set_data(np.rot90(data.asarray()))
        ax.add_artist(bbox_image)
        
    show(coarse_data)
    show(fine_data)

# %%

class MaskingOperator(odl.Operator):

    def __init__(self, space, min_pt, max_pt):

        super().__init__(domain=space, range=space, linear=True)
        self.min_pt = min_pt
        self.max_pt = max_pt

    def _call(self, x, out):
        # TODO: find better way of getting the indices
        idx_min_flt = ((self.min_pt - self.domain.min_pt) /
                       self.domain.cell_sides)
        idx_max_flt = ((self.max_pt - self.domain.min_pt) /
                       self.domain.cell_sides)

        # to deal with coinciding boundaries we introduce an epsilon tolerance
        epsilon = 1e-6
        idx_min = np.floor(idx_min_flt - epsilon).astype(int)
        idx_max = np.ceil(idx_max_flt + epsilon).astype(int)
        
        coeffs = lambda d: (1.0 - (idx_min_flt[d] - idx_min[d]),
                            1.0 - (idx_max[d] - idx_max_flt[d]))
        
        # we need an extra level of indirection for capturing `d` inside
        # the lambda
        def fn_pair(d):
            return (lambda x: x * coeffs(d)[0], lambda x: x * coeffs(d)[1])
            
        boundary_scale_fns = [fn_pair(d) for d in range(x.ndim)]
        
        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        slc_inner = tuple(slice(imin + 1, imax - 1) for imin, imax in
                          zip(idx_min, idx_max))
        
        out.assign(x)
        with writable_array(out) as out_arr:
            out_arr[slc_inner] = 0
            odl.util.numerics.apply_on_boundary(out_arr[slc],
                                                boundary_scale_fns,
                                                only_once=False,
                                                out=out_arr[slc])
            odl.util.numerics.apply_on_boundary(out_arr[slc],
                                                lambda x: 1.0 - x,
                                                only_once=True,
                                                out=out_arr[slc])

    @property
    def adjoint(self):
        return self


# %%
coarse_discr = odl.uniform_discr([-10, -10], [10, 10], [50, 50])
fine_min = [-1, -1]
fine_max = [1, 1]
fine_discr = odl.uniform_discr(fine_min, fine_max, [100, 100])

angle_partition = odl.uniform_partition(0, np.pi, 180)
det_partition = odl.uniform_partition(-15, 15, 3000)

geometry = odl.tomo.Parallel2dGeometry(angle_partition, det_partition,
                                       det_init_pos=[20, 0])

ray_trafo_coarse = odl.tomo.RayTransform(coarse_discr, geometry,
                                         impl='astra_cpu')

coarse_mask = MaskingOperator(coarse_discr, fine_min, fine_max)
masked_ray_trafo_coarse = ray_trafo_coarse * coarse_mask

ray_trafo_fine = odl.tomo.RayTransform(fine_discr, geometry,
                                       impl='astra_cpu')

pspace_ray_trafo = odl.ReductionOperator(masked_ray_trafo_coarse,
                                         ray_trafo_fine)
pspace = pspace_ray_trafo.domain
phantom_c = odl.phantom.shepp_logan(coarse_discr, modified=True)
phantom_f = odl.phantom.shepp_logan(fine_discr, modified=True)
phantom = pspace.element([phantom_c, phantom_f])

data = pspace_ray_trafo([phantom_c, phantom_f])
data.show('data')

noisy_data = data + odl.phantom.white_noise(ray_trafo_coarse.range, stddev=0.1)
noisy_data.show('noisy data')

reco = pspace_ray_trafo.domain.zero()

# %%
# CG reconstruction
callback = odl.solvers.CallbackShow(display_step=5)
odl.solvers.conjugate_gradient_normal(pspace_ray_trafo, reco, data, niter=20,
                                      callback=callback)

# %%
fine_grad = odl.Gradient(fine_discr, pad_mode='order1')

# Differentiable part, build as ||. - g||^2 o P
data_func = odl.solvers.L2NormSquared(
    pspace_ray_trafo.range).translated(noisy_data) * pspace_ray_trafo
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
box_constr = odl.solvers.IndicatorBox(fine_discr,
                                      np.min(phantom_f), np.max(phantom_f))
f = odl.solvers.ZeroFunctional(pspace)

# %%
# eta^-1 is the Lipschitz constant of the smooth functional gradient
ray_trafo_norm = 1.1 * odl.power_method_opnorm(pspace_ray_trafo,
                                               xstart=phantom, maxiter=2)
print('norm of the ray transform: {}'.format(ray_trafo_norm))
eta = 1 / (2 * ray_trafo_norm ** 2 + 2 * reg_param_1)
print('eta = {}'.format(eta))
grad_norm = 1.1 * odl.power_method_opnorm(fine_grad, xstart=phantom_f,
                                          maxiter=4)
print('norm of the gradient: {}'.format(grad_norm))

# %%
# tau and sigma are like step sizes
sigma = 5e-3
tau = 2 * sigma
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
x = pspace.zero()  # starting point
odl.solvers.forward_backward_pd(x, f=f, g=nonsmooth_funcs, L=lin_ops,
                                h=smooth_func,
                                tau=tau, sigma=[sigma], niter=60,
                                callback=callback)

# %%

coarse_grid = odl.uniform_discr([-2.5, -2.5], [2.5, 2.5], [5, 5])
xmin = [-1.2, -1.2]
xmax = [1, 1]
fine_grid = odl.uniform_discr(xmin, xmax, [5, 5])
show_both(coarse_grid.zero(), fine_grid.one())