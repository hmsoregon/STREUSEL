from . import geometry_manipulation_functions as gm
from . import process_functions as pf 
from . import math_functions as mf 
from tqdm import tqdm
import pandas as pd 
import numpy as np 
import pickle
import math
import os
import ase
import scipy.ndimage

# -------------------------------------------------------------------------------------------------------
# Defines object type Molecule >> can handle Gaussian .cube output file
# -------------------------------------------------------------------------------------------------------

class Molecule:
    """ Class for properties of single molecules """

    def __init__(self, file):
        """ read in cube file using ase, and assign material properties """
        potential, ngx, ngy, ngz, atoms, lattice, coords, species, origin, resolution = pf.read_cube_file(file)
        self.pot_grid = potential
        self.ngs = [ngx, ngy, ngz]
        self.atoms = atoms
        self.lattice = lattice
        self.coords = coords
        self.species = species
        # get 'n' the scaling factor for PV and SA calcs
        self.n = pf.get_n(species, 'None')
        # self.atom_symbols = symbols
        self.coords_and_atoms = gm.combine_coordsANDatoms(coords, species)
        self.vecs = lattice
        self.origin = origin
        self.res = resolution
        print(atoms)

    def read_cube_file(self, filename):
        """ reads in Gaussian cube format using ASE """
        potential, atoms = ase.io.cube.read_cube_data(filename)
        vector_a = np.linalg.norm(atoms.cell[0])
        vector_b = np.linalg.norm(atoms.cell[1])
        vector_c = np.linalg.norm(atoms.cell[2])
        NGX = len(potential)
        NGY = len(potential[0])
        NGZ = len(potential[0][0])
        resolution_x = vector_a/NGX
        resolution_y = vector_b/NGY
        resolution_z = vector_c/NGZ
        coords = atoms.get_positions()
        atom_types = atoms.get_chemical_symbols()

        with open(filename, 'r') as cube:
            lines = cube.readlines()
        origin = [float(i) for i in lines[2].split()[1:-1]]

        return potential, NGX, NGY, NGZ, atoms, [vector_a, vector_b, vector_c], coords, atom_types, origin, [resolution_x, resolution_y, resolution_z]

    def get_CenterOfMass(self):
        """ Returns the NG* values of the Center of Mass """

        atomic_mass = {'H':1.01, 'He':4.00, 'Li':6.94, 'Be':9.01, 'B':10.81, 'C':12.01,
                    'N':14.01, 'O':16.00, 'F':19.00, 'Ne':20.18, 'Na':22.99, 'Mg':24.31,
                    'Al':26.98, 'Si':28.09, 'P':30.97, 'S':32.07, 'Cl':35.45, 'Ar':39.95,
                    'K':39.10, 'Ca':40.08, 'Sc':44.96, 'Ti':47.87, 'V':50.94, 'Cr':52.00,
                    'Mn':54.94, 'Fe':55.85, 'Co':58.93, 'Ni':58.69, 'Cu':63.55, 'Zn':65.39,
                    'Ga':69.72, 'Ge':72.61, 'As':74.92, 'Se':78.96, 'Br':79.90, 'Kr':83.80,
                    'Rb':85.47, 'Sr':87.62, 'Y':88.91, 'Zr':91.22, 'Nb':92.91, 'Mo':95.94,
                    'Tc':98.00, 'Ru':101.07, 'Rh':102.91, 'Pd':106.42, 'Ag':107.87,
                    'Cd':112.41, 'In':114.82, 'Sn':118.71, 'Sb':121.76, 'Te':127.60,
                    'I':126.90, 'Xe':131.29, 'Cs':132.91, 'Ba':137.33, 'La':138.91,
                    'Ce':140.12, 'Pr':140.91, 'Nd':144.24, 'Pm':145.00, 'Sm':150.36,
                    'Eu':151.96, 'Gd':157.25, 'Tb':158.93, 'Dy':162.50, 'Ho':164.93,
                    'Er':167.26, 'Tm':168.93, 'Yb':173.04, 'Lu':174.97, 'Hf':178.49,
                    'Ta':180.95, 'W':183.84, 'Re':186.21, 'Os':190.23, 'Ir':192.22,
                    'Pt':195.08, 'Au':196.97, 'Hg':200.59, 'Tl':204.38, 'Pb':207.2,
                    'Bi':208.98, 'Po':209.00, 'At':210.00, 'Rn':222.00, 'Fr':223.00,
                    'Ra':226.00, 'Ac':227.00, 'Th':232.04, 'Pa':231.04, 'U':238.03,
                    'Np':237.00, 'Pu':244.00, 'Am':243.00, 'Cm':247.00, 'Bk':247.00,
                    'Cf':251.00, 'Es':252.00, 'Fm':257.00, 'Md':258.00, 'No':259.00,
                    'Lr':262.00, 'Rf':261.00, 'Db':262.00, 'Sg':266.00, 'Bh':264.00,
                    'Hs':269.00, 'Mt':268.00}
        total_mass = 0
        comx = 0
        comy = 0
        comz = 0

        for index, row in self.coords_and_atoms.iterrows():
            atom_mass = atomic_mass[str(row['atom'])]
            comx += atom_mass * row['x']
            comy += atom_mass * row['y']
            comz += atom_mass * row['z']
            total_mass += atom_mass
        comx = comx/total_mass
        comy = comy/total_mass
        comz = comz/total_mass
        com = pd.DataFrame(data=[[comx, comy, comz]])
        com.columns = ['x', 'y', 'z']
        return com

    def get_efield(self):
        """ Derives the electric field from the electrostatic potential """
        n = self.n
        ng = self.ngs
        vecs = self.vecs
        vol_per_cube = (vecs[0]/ng[0]) * (vecs[1]/ng[1]) * (vecs[2]/ng[2])

        pgrid = self.pot_grid
        res = [vecs[0]/ng[0], vecs[1]/ng[1], vecs[2]/ng[2]]

        grad_x, grad_y, grad_z = np.gradient(pgrid[:,:,:], res[0], res[1], res[2])

        # xy = np.multiply(grad_x, grad_y)
        # grad_mag = np.multiply(xy, grad_z)
        grad_mag = mf.grad_magnitude(grad_x, grad_y, grad_z)
        print(grad_mag.shape)
        self.efield = grad_mag

    def longest_vector(self):
      """ determines the max distance from the center of mass to the surface """
      vac = self.sijk
      vecs = self.vecs
      ngs = self.ngs
      # dist_matrix = np.zeros(shape=(len(vac), 7))
      # row = 0
      distances = []

      for p1 in tqdm(vac):
        for p2 in vac:
          distances.append(gm.distance(p1[0],p2[0]))
      lv = max(distances)
      lvx = (vecs[0]/ngs[0]) * lv * 0.529
      lvy = (vecs[1]/ngs[1]) * lv * 0.529
      lvz = (vecs[2]/ngs[2]) * lv * 0.529

      self.l_vector = np.average([lvx, lvy, lvz])

    def sample_density(self, method):
      """ Samples the electron density """
      dens = self.pot_grid
      ngs = self.ngs
      cv = method

      vacuum = []
      non_vacuum = []
      surf_ijk = []
      surface = []
      vac_ijk = []

      v = 0
      n = 0
      get_first_rise = 0

      for i in tqdm(range(ngs[0]-1)):
        for j in range(ngs[1]-1):
          for k in range(ngs[2]-1):
            if dens[i,j,k] <= method:
              vacuum.append(dens[i,j,k])
              vac_ijk.append([[i,j,k]])
              v += 1
              n = 0
            else:
              non_vacuum.append(dens[i,j,k])
              n += 1
              v = 0
            if (n+v) == 1:
              surface.append(dens[i,j,k])
              surf_ijk.append([[i,j,k]])
      vecs = self.vecs
      vol_per_cube = (vecs[0]/ngs[0]) * (vecs[1]/ngs[1]) * (vecs[2]/ngs[2])
      volume = vol_per_cube * len(non_vacuum) * np.power(0.529, 3)

      self.vol = volume
      #self.vac = vacuum
      #self.vijk = vac_ijk
      #self.surface = surface
      #self.sijk = surf_ijk

    def sample_test(self):
        efield = self.efield
        ngs = self.ngs
        cv = 1e-5
        vacuum = []
        non_vacuum = []
        surf_ijk = []
        surface = []

        v = 0
        n = 0
        get_first_rise = 0
        for i in tqdm(range(ngs[0]-1)):
          for j in range(ngs[1]-1):
            for k in range(ngs[2]-1):
              if np.absolute(efield[i,j,k] - efield[i+1, j+1, k+1]) <= cv:
                vacuum.append(efield[i,j,k])
                v += 1
                n = 0
              else:
                non_vacuum.append(efield[i,j,k])
                n += 1
                v = 0
              if (n+v) == 2:
                if get_first_rise == 0:
                  surface.append(efield[i,j,k])
                  surf_ijk.append([[i,j,k]])
                  needed_mag = efield[i,j,k]
                  get_first_rise += 1
                elif np.absolute(efield[i,j,k] - needed_mag) <= 1e-10:
                  surface.append(efield[i,j,k])
                  surf_ijk.append([[i,j,k]])
                  get_first_rise = 0

        vecs = self.vecs
        vol_per_cube = (vecs[0]/ngs[0]) * (vecs[1]/ngs[1]) * (vecs[2]/ngs[2])
        volume = vol_per_cube * len(non_vacuum) * np.power(0.529, 3)

        self.vol = volume
        self.nvac = non_vacuum
        self.surf = surface
        self.sijk = surf_ijk
        self.vac = vacuum

    def sample_efield(self):
        """ samples the electric field and bins the vac, nvac, and surface cubes """
        efield = self.efield
        ngs = self.ngs
        vecs = self.vecs
        cv = 1e-5
        vacuum = []
        non_vacuum = []
        surface = []
        vac_ijk = []
        nvac_ijk = pd.DataFrame()
        surf_ijk = []
        vol_per_cube = (vecs[0]/ngs[0]) * (vecs[1]/ngs[1]) * (vecs[2]/ngs[2])
        v = 0 
        n = 0

        efield_view = efield[:-1,:-1,:-1]
        efield_diff = efield_view - efield[1:, 1:, 1:]
        np.absolute(efield_diff, out=efield_diff)
        is_vacuum = efield_diff < cv
        is_non_vacuum = np.logical_not(is_vacuum)
        vacuum = efield_view[is_vacuum]
        non_vacuum = efield_view[is_non_vacuum]
        nvac_ijk = np.argwhere(is_non_vacuum)
        weights = np.zeros((3,3,3))
        weights[1,1,2] = 1
        weights[1,1,0] = 1
        weights[:,:,1] = [
                [0,1,0],
                [1,0,1],
                [0,1,0]
                ]
        convolution_result = scipy.ndimage.convolve(
                input=is_vacuum.view(np.int8),
                weights = weights,
                mode='wrap'
                )
        np.multiply(
                convolution_result,
                is_non_vacuum.view(np.int8),
                out=convolution_result,
                )
        surface_mask = convolution_result > 0
        surf_ijk = np.argwhere(surface_mask)
        mk = (vecs[0]/ngs[0]) * (vecs[1]/ngs[1])
        sarea = np.sum(surface_mask) * mk
        sarea2 = np.sum(convolution_result) * mk
        print('surface areas ', sarea, sarea2)
        print('volume', np.sum(is_non_vacuum)*vol_per_cube)
        """
        vc = 0
        nc = 0
        get_first_rise = 0
        for i in tqdm(range(ngs[0] - 1)):
            for j in range(ngs[1] - 1):
                count = 1
                for k in range(ngs[2] - 1):
                    if np.absolute(efield[i,j,k] - efield[i + 1, j + 1, k + 1]) <= cv:
                        vacuum.append(efield[i,j,k])
                        v += 1
                        n = 0
                    else:
                        non_vacuum.append(efield[i,j,k])
                        n += 1
                        v = 0
                    if n/count != 1.0 and v/count != 1.0:
                        if get_first_rise == 0:
                            surface.append(efield[i,j,k])
                            surf_ijk.append([[i, j, k]])
                            needed_mag = efield[i,j,k]
                            n = 0
                            v = 0
                            count = 0
                            get_first_rise += 1
                        elif np.absolute(efield[i,j,k] - needed_mag) <= 1e-10:
                            surface.append(efield[i,j,k])
                            surf_ijk.append([[i,j,k]])
                            n = 0
                            v = 0
                            count = 0
                    count += 1
                get_first_rise = 0
        """
        vecs = self.vecs
        vol_per_cube = (vecs[0]/ngs[0]) * (vecs[1]/ngs[1]) * (vecs[2]/ngs[2])
        ubound_vol = vol_per_cube * len(non_vacuum) # * np.power(0.529,3)

        self.vol = ubound_vol
        #self.nvac = non_vacuum
        self.surf = surface
        self.sijk = surf_ijk
        #self.vac = vacuum

    def get_radii(self):
        """ calculates the min and max radius (between COM and surface"""
        def get_com():
            atomic_mass = {'H':1.01, 'He':4.00, 'Li':6.94, 'Be':9.01, 'B':10.81, 'C':12.01,
                        'N':14.01, 'O':16.00, 'F':19.00, 'Ne':20.18, 'Na':22.99, 'Mg':24.31,
                        'Al':26.98, 'Si':28.09, 'P':30.97, 'S':32.07, 'Cl':35.45, 'Ar':39.95,
                        'K':39.10, 'Ca':40.08, 'Sc':44.96, 'Ti':47.87, 'V':50.94, 'Cr':52.00,
                        'Mn':54.94, 'Fe':55.85, 'Co':58.93, 'Ni':58.69, 'Cu':63.55, 'Zn':65.39,
                        'Ga':69.72, 'Ge':72.61, 'As':74.92, 'Se':78.96, 'Br':79.90, 'Kr':83.80,
                        'Rb':85.47, 'Sr':87.62, 'Y':88.91, 'Zr':91.22, 'Nb':92.91, 'Mo':95.94,
                        'Tc':98.00, 'Ru':101.07, 'Rh':102.91, 'Pd':106.42, 'Ag':107.87,
                        'Cd':112.41, 'In':114.82, 'Sn':118.71, 'Sb':121.76, 'Te':127.60,
                        'I':126.90, 'Xe':131.29, 'Cs':132.91, 'Ba':137.33, 'La':138.91,
                        'Ce':140.12, 'Pr':140.91, 'Nd':144.24, 'Pm':145.00, 'Sm':150.36,
                        'Eu':151.96, 'Gd':157.25, 'Tb':158.93, 'Dy':162.50, 'Ho':164.93,
                        'Er':167.26, 'Tm':168.93, 'Yb':173.04, 'Lu':174.97, 'Hf':178.49,
                        'Ta':180.95, 'W':183.84, 'Re':186.21, 'Os':190.23, 'Ir':192.22,
                        'Pt':195.08, 'Au':196.97, 'Hg':200.59, 'Tl':204.38, 'Pb':207.2,
                        'Bi':208.98, 'Po':209.00, 'At':210.00, 'Rn':222.00, 'Fr':223.00,
                        'Ra':226.00, 'Ac':227.00, 'Th':232.04, 'Pa':231.04, 'U':238.03,
                        'Np':237.00, 'Pu':244.00, 'Am':243.00, 'Cm':247.00, 'Bk':247.00,
                        'Cf':251.00, 'Es':252.00, 'Fm':257.00, 'Md':258.00, 'No':259.00,
                        'Lr':262.00, 'Rf':261.00, 'Db':262.00, 'Sg':266.00, 'Bh':264.00,
                        'Hs':269.00, 'Mt':268.00}
            total_mass = 0
            comx = 0
            comy = 0
            comz = 0

            for index, row in self.coords_and_atoms.iterrows():
                atom_mass = atomic_mass[str(row['atom'])]
                comx += atom_mass * row['x']
                comy += atom_mass * row['y']
                comz += atom_mass * row['z']
                total_mass += atom_mass
            comx = comx/total_mass
            comy = comy/total_mass
            comz = comz/total_mass
            com = pd.DataFrame(data=[[comx, comy, comz]])
            com.columns = ['x', 'y', 'z']
            return com
        vecs = self.vecs
        ngs = self.ngs
        com = get_com()
        print(com)
        com_cube = [int(ngs[0]/2), int(ngs[1]/2), int(ngs[2]/2)]
        print(com_cube)
        sijk = self.sijk
        sijk2 = []
        for p1 in sijk:
            if com_cube == p1[0]:
                pass
            elif p1[0][0] == 0 or p1[0][1] == 0 or p1[0][2] == 0:
                pass
            else:
                sijk2.append(p1)
        arr = np.zeros(shape=(len(sijk2), 4))
        row = 0
        for p1 in sijk2:
            if p1[0][0] == 0 or p1[0][1] == 0 or p1[0][2] == 0:
                print(p1)
            else:
                arr[row, 0] = p1[0][0]
                arr[row, 1] = p1[0][1]
                arr[row, 2] = p1[0][2]
                arr[row, 3] = gm.distance(p1[0], com_cube)
                row += 1

        try:
            abs_min_length = np.min(arr[:,3])
            abs_xminlen = (vecs[0]/ngs[0]) * abs_min_length * 0.529
            abs_yminlen = (vecs[1]/ngs[1]) * abs_min_length * 0.529
            abs_zminlen = (vecs[2]/ngs[2]) * abs_min_length * 0.529

            abs_max_length = np.max(arr[:,3])
            abs_xmaxlen = (vecs[0]/ngs[0]) * abs_max_length * 0.529
            abs_ymaxlen = (vecs[1]/ngs[1]) * abs_max_length * 0.529
            abs_zmaxlen = (vecs[2]/ngs[2]) * abs_max_length * 0.529

            self.min_rad = np.average([abs_xminlen, abs_yminlen, abs_zminlen])
            self.max_rad = np.average([abs_xmaxlen, abs_ymaxlen, abs_zmaxlen])
        except ValueError:
            self.min_rad = 0
            self.max_rad = 0
        # min_rad = np.average([abs_xminlen, abs_yminlen, abs_zminlen])
        # max_rad = np.average([abs_xmaxlen, abs_ymaxlen, abs_zmaxlen])

        # return min_rad, max_rad

