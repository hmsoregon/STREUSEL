import sys
sys.path.insert(0,'path-to-STREUSEL')
from vasp_object import *

# Define material object and read in LOCPOT
mat = Material('LOCPOT-file-name')

# Derive electric field from electrostatic potential values in LOCPOT
mat.get_efield()

# Find the surface, material, and vacuum cubes within the LOCPOT
mat.sample_field()

# Calculate the volume
mat.calc_volume('MOF_pore')

# Calculate the surface area
mat.calc_surfacearea()

print(mat.vol, mat.surf_area)

