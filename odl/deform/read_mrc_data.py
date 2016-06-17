import odl
import numpy as np

# Change this
directory = '''/home/hkohr/SciData/Electron_Tomography/Single_Axis/\
Simulated/RNA_Polymerase_II/'''
filename = 'ET_data_one_RNA_nonoise.mrc'
reader = odl.tomo.data.MRCFileReader(directory + filename)

data = reader.read_data()

shape = reader.data_shape
csides = reader.cell_sides_angstrom
extent = csides * shape

space = odl.uniform_discr(-extent / 2, extent / 2, shape)
data_elem = space.element(data)

data_elem.show(indices=np.s_[..., shape[-1] // 2])
