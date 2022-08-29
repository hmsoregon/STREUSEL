import streusel
from streusel.vasp_object import *

# Define material object and read in LOCPOT
mat = Material('Ne_LOCPOT')

# Derive electric field from electrostatic potential values in LOCPOT
mat.get_efield()

# Find the surface, material, and vacuum cubes within the LOCPOT
mat.sample_field()

# Calculate the volume
mat.calc_volume('molecule')

print(mat.vol) # , mat.surf_area)

