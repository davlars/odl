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
        # TODO: allow ignoring axes
        # TODO: allow masking array
        # TODO: sanitize input
        self.min_pt = min_pt
        self.max_pt = max_pt

    def _call(self, x, out):

        idx_min = self.domain.index(self.min_pt)
        idx_max = self.domain.index(self.max_pt)

        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        out.assign(x)
        with writable_array(out) as out_arr:
            out_arr[slc] = 0

    @property
    def adjoint(self):
        return self


# %% MultigridGradient

class MultiGridGradient(odl.DiagonalOperator):

    def __init__(self, coarse_discr, fine_discr, method='forward',
                 pad_mode='constant', pad_const=0):

        self.coarse_grad = odl.Gradient(coarse_discr, method=method,
                                        pad_mode=pad_mode, pad_const=pad_const)

        # TODO: handle case when fine discr touches boundary
        # TODO: change pad_mode, this is just for debugging
        self.fine_grad = odl.Gradient(fine_discr, method=method,
                                      pad_mode='constant', pad_const=0)

        super().__init__(self.coarse_grad, self.fine_grad)

    @property
    def coarse_discr(self):
        return self.domain[0]

    @property
    def fine_discr(self):
        return self.domain[1]

    def overlaps(self, coarse_arr, fine_arr):
        # TODO: generalize to non-matching overlaps
        # TODO: handle corner cases of small inserts etc.
        coarse_cell_sides = self.coarse_discr.cell_sides
        fine_cell_sides = self.fine_discr.cell_sides
        xmin, xmax = self.coarse_discr.min_pt, self.coarse_discr.max_pt
        ymin, ymax = self.fine_discr.min_pt, self.fine_discr.max_pt

        ymin_idx = self.coarse_discr.index(ymin)
        ymin_idx_f = self.coarse_discr.index(ymin, floating=True)
        ymax_idx = self.coarse_discr.index(ymax)
        ymax_idx_f = self.coarse_discr.index(ymax, floating=True)

        print('ymin:', ymin)
        print('ymin_idx:', ymin_idx)
        print('ymin_idx_f:', ymin_idx_f)
        print('ymax:', ymax)
        print('ymax_idx:', ymax_idx)
        print('ymax_idx_f:', ymax_idx_f)

        coarse_cvecs = self.coarse_discr.partition.coord_vectors
        min_next = tuple(cvec[i + 1]
                         for cvec, i in zip(coarse_cvecs, ymin_idx))
        min_before = tuple(cvec[i]
                           for cvec, i in zip(coarse_cvecs, ymin_idx))
        max_before = tuple(cvec[i - 1]
                           for cvec, i in zip(coarse_cvecs, ymax_idx))
        max_next = tuple(cvec[i]
                         for cvec, i in zip(coarse_cvecs, ymax_idx))

        print('min_before:', np.array(min_before))
        print('min_next:', np.array(min_next))
        print('max_before:', np.array(max_before))
        print('max_next:', np.array(max_next))

        # TODO: take one extra if available
        min_next_fine_idx = self.fine_discr.index(min_next)
        max_before_fine_idx = self.fine_discr.index(max_before)

        print('min_next_fine_idx:', min_next_fine_idx)
        print('max_before_fine_idx:', max_before_fine_idx)

        left_avg_arrays, right_avg_arrays = [], []
        ndim = self.coarse_discr.ndim
        # TODO: perhaps better to make cut-out spaces and use integration
        # there
        for i, (min_i, max_i) in enumerate(zip(min_next_fine_idx,
                                               max_before_fine_idx)):
            slc_l = [slice(None)] * ndim
            slc_l[i] = slice(min_i)
            left_avg_arrays.append(np.mean(fine_arr[slc_l], axis=i))
            print(left_avg_arrays[-1])

            slc_r = [slice(None)] * ndim
            slc_r[i] = slice(max_i, None)
            right_avg_arrays.append(np.mean(fine_arr[slc_r], axis=i))
            print(right_avg_arrays[-1])

    def _call(self, x, out):
        pass

    @property
    def adjoint(self):
        return None


# %% Set up the operators and phantoms

coarse_discr = odl.uniform_discr([-10, -10], [10, 10], [50, 50],
                                 dtype='float32')
fine_min = [1, 1]
fine_max = [2, 2]
fine_discr = odl.uniform_discr(fine_min, fine_max, [100, 100], dtype='float32')

multi_grad = MultiGridGradient(coarse_discr, fine_discr)

coarse_arr = coarse_discr.zero().asarray()
fine_arr = fine_discr.one().asarray()

multi_grad.overlaps(coarse_arr, fine_arr)