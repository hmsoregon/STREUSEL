import numpy as np 

def c2f(x):
    """ 
    Converts 'x' to a float 
    """
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
    """
    Converts the separate gradient magnitudes to a single magnitude
    Args:
        gx/y/z : fields in x y and z directions 2D array
    Returns:
        grad_mag : gradient of fields at each point
    """

    out = np.square(gx)
    out += np.square(gy)
    out += np.square(gz)
        
    return np.sqrt(out, out=out)

