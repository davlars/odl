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

        # TODO: find better way of getting the indices
        idx_min_flt = ((self.min_pt - self.domain.min_pt) /
                       self.domain.cell_sides)
        idx_max_flt = ((self.max_pt - self.domain.min_pt) /
                       self.domain.cell_sides)

        # Fix for numerical instability
        idx_min = idx_min_flt.astype(int)
        idx_max = idx_max_flt.astype(int)

        # TODO: assign proper weights to the cut-out region
        for j in range(len(idx_min_flt)):
            if not np.isclose(idx_min_flt[j], idx_min[j]):
                idx_min[j] = int(np.floor(idx_min_flt[j]))
            if not np.isclose(idx_max_flt[j], idx_max[j]):
                idx_max[j] = int(np.ceil(idx_max_flt[j]))

        # FIXME: something wrong here, lower index too low

        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        out.assign(x)
        with writable_array(out) as out_arr:
            out_arr[slc] = 0

    @property
    def adjoint(self):
        return self


# %% MultigridGradient

class MultiGridGradient(odl.ProductSpaceOperator):

    def __init__(self, coarse_discr, fine_discr, method='forward',
                 pad_mode='constant', pad_const=0):
        pass

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

