#!/usr/bin/env python
# encoding: utf-8
"""
Foundation classes for the `ddmatch` library.

Documentation guidelines are available `here <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_.

Created by Klas Modin on 2014-11-03.
"""

import numpy as np 
import matplotlib.pyplot as plt

class Presentation(object):
	"""
	Presentation of a matching process computed using any of the `ddmatch`
	core classes.

	The visualization depends on the `matplotlib` library.
	"""
	def __init__(self, arg):
		super(Presentation, self).__init__()
		self.arg = arg
		


if __name__ == '__main__':
	pass
