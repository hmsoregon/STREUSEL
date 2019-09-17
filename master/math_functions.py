import numpy as np 

def c2f(x):
	""" Converts 'x' to a float """
	try:
		x = int(x)
		return x
	except ValueError:
		try:
			x = float(x)
			return x
		except ValueError:
			return x

def grad_magnitude(gx, gy, gz):
	"""Converts the separate gradient magnitudes to a single magnitude
Args:
	gx/y/z : fields in x y and z directions 2D array
Returns:
	grad_mag : gradient of fields at each point"""
	grad_mag = gx
	for i in range(gx.shape[0]):
		for j in range(gy.shape[1]):
			for k in range(gz.shape[2]):
				grad_mag[i,j,k] = np.sqrt(gx[i,j,k]**2+gy[i,j,k]**2+gz[i,j,k]**2)

	return grad_mag