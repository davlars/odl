# Copyright 2014-2017 The ODL development group
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

"""Simple iterative type optimization schemes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


from odl.solvers.util import ConstantLineSearch, BacktrackingLineSearch


__all__ = ('conjugate_gradient_nonlinear',)


def conjugate_gradient_nonlinear(f, x, line_search=1.0, maxiter=1000, nreset=0,
                                 tol=1e-16, beta_method='FR',
                                 callback=None):
    """Conjugate gradient for nonlinear problems.

    Notes
    -----
    This is a general and optimized implementation of the nonlinear conjguate
    gradient method for solving a general unconstrained optimization problem

        :math:`\min f(x)`

    for a differentiable function
    :math:`f: \mathcal{X}\\to \mathbb{R}` on a Hilbert space
    :math:`\mathcal{X}`. It does so by finding a zero of the gradient

        :math:`\\nabla f: \mathcal{X} \\to \mathcal{X}`.

    The method is described in a
    `Wikipedia article
    <https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method>`_.

    Parameters
    ----------
    f : `Functional`
        Functional with ``f.gradient``.
    x : ``op.domain`` element
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
    line_search : float or `LineSearch`, optional
        Strategy to choose the step length. If a float is given, uses it as a
        fixed step length.
    maxiter : int
        Maximum number of iterations to perform.
    nreset : int, optional
        Number of times the solver should be reset. Default: no reset.
        fixed step length. Default: `BacktrackingLineSearch`
    tol : float, optional
        Tolerance that should be used for terminating the iteration.
    beta_method : {'FR', 'PR', 'HS', 'DY'}
        Method to calculate ``beta`` in the iterates.
        * ``'FR'`` : Fletcher-Reeves
        * ``'PR'`` : Polak-Ribiere
        * ``'HS'`` : Hestenes-Stiefel
        * ``'DY'`` : Dai-Yuan
    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate

    See Also
    --------
    bfgs_method : Quasi-newton solver for the same problem
    conjugate_gradient : Optimized solver for linear and symmetric case
    conjugate_gradient_normal : Equivalent solver but for linear case
    """

    if x not in f.domain:
        raise TypeError('`x` {!r} is not in the domain of `f` {!r}'
                        ''.format(x, f.domain))

    if line_search is None:
        line_search = BacktrackingLineSearch(f, estimate_step=True)
    elif not callable(line_search):
        line_search = ConstantLineSearch(line_search)

    for _ in range(nreset + 1):
        # First iteration is done without beta
        dx = -f.gradient(x)
        dir_derivative = -dx.inner(dx)
        if abs(dir_derivative) < tol:
            return
        a = line_search(x, dx, dir_derivative)
        x.lincomb(1, x, a, dx)  # x = x + a * dx

        dx_old = dx
        s = dx  # for 'HS' and 'DY' beta methods

        for _ in range(maxiter // (nreset + 1)):
            # Compute dx as -grad f
            dx, dx_old = -f.gradient(x), dx

            # Calculate "beta"
            if beta_method == 'FR':
                beta = dx.inner(dx) / dx_old.inner(dx_old)
            elif beta_method == 'PR':
                beta = dx.inner(dx - dx_old) / dx_old.inner(dx_old)
            elif beta_method == 'HS':
                beta = dx.inner(dx - dx_old) / s.inner(dx - dx_old)
            elif beta_method == 'DY':
                beta = dx.inner(dx) / s.inner(dx - dx_old)
            else:
                raise ValueError('unknown ``beta_method``')

            # Reset beta if negative.
            beta = max(0, beta)

            # Update search direction
            s.lincomb(1, dx, beta, s)  # s = dx + beta * s

            # Find optimal step along s
            dir_derivative = -dx.inner(s)
            if abs(dir_derivative) < tol:
                return
            a = line_search(x, s, dir_derivative)

            # Update position
            x.lincomb(1, x, a, s)  # x = x + a * s

            if callback is not None:
                callback(x)
