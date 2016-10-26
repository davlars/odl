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

        # TODO: assign proper weights to the cut-out region
        idx_min = np.floor(idx_min_flt).astype(int)
        idx_max = np.ceil(idx_max_flt).astype(int)

        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        out.assign(x)
        with writable_array(out) as out_arr:
            out_arr[slc] = 0

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

data = pspace_ray_trafo([phantom_c, phantom_f])
data.show('data')

noisy_data = data + odl.phantom.white_noise(ray_trafo_coarse.range, stddev=0.1)
noisy_data.show('noisy data')

reco = pspace_ray_trafo.domain.zero()

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

# Non-differentiable part composed with linear operators
reg_param = 7e-3
nonsmooth_func = reg_param * odl.solvers.L1Norm(fine_grad.range)

# Assemble into lists (arbitrary number can be given)
comp_proj_1 = odl.ComponentProjection(pspace, 1)
lin_ops = [fine_grad * comp_proj_1]
nonsmooth_funcs = [nonsmooth_func]

