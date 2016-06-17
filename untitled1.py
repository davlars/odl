# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:49:31 2016

@author: chong
"""

# Test run
import odl
import numpy as np

# Set geometry parameters
volumeSize = np.array([10, 10, 10]) #mm?
volumeOrigin = np.array([-5, -5, -5.0]) #mm?
nVoxels = np.array([100,100,100])
detectorSize = np.array([30.0, 3.0])
nPixels = [300, 30]
detectorOrigin = -detectorSize/2

# Discrete reconstruction space
discr_reco_space = odl.uniform_discr(volumeOrigin,volumeOrigin + volumeSize,
                                     nVoxels, dtype='float32')

#Define projection geometry
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)

# Detector: uniformly sampled, n = nPixels, min = detectorOrigin, max = detectorOrigin+detectorSize
detector_partition = odl.uniform_partition(detectorOrigin, detectorOrigin+detectorSize, nPixels)

# Spiral has a pitch of pitch_mm, we run n_turns rounds (due to max angle = 8 * 2 * pi)
geom = odl.tomo.HelicalConeFlatGeometry(angle_partition, 
                                        detector_partition, 
                                        src_radius=250, 
                                        det_radius=250,
                                        pitch=0,
                                        pitch_offset=0)

A = odl.tomo.RayTransform(discr_reco_space, geom, impl='astra_cuda')
# A = odl.trafos.FourierTransform(discr_reco_space) also fails
phantom = odl.phantom.cuboid(A.domain)
rhs = A(phantom)

# Reconstruct
partialPrintIter = odl.solvers.CallbackPrintIteration() #Print iterations
partialShowIter = odl.solvers.CallbackShow(coords=[None, None, 0]) #Show parital reconstructions

# THIS FAILS

#partialIter = odl.solvers.util.PrintIterationPartial()
x = discr_reco_space.zero()
odl.solvers.conjugate_gradient_normal(A, x, rhs, niter=1000, callback = partialPrintIter & partialShowIter)

# THESE WORK JUST FINE
#odl.solvers.conjugate_gradient(A.adjoint * A, x, A.adjoint(rhs), niter=1000, callback = partialPrintIter & partialShowIter)
#odl.solvers.landweber(A, x, rhs, niter=1000, omega=0.1, callback = partialPrintIter & partialShowIter)