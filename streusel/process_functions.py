from . import math_functions as mf
from ase.visualize import view
import pandas as pd
import numpy as np
import ase.io.cube
import pickle

def read_cube_file(filename):

    potential, atoms = ase.io.cube.read_cube_data(filename)
    vectors = np.linalg.norm(atoms.cell, axis=1)
    NGX, NGY, NGZ = potential.shape
    resolution = vectors / potential.shape
    coords = atoms.get_positions()
    atom_types = atoms.get_chemical_symbols()

    with open(filename, 'r') as cube:
        lines = cube.readlines()
    origin = [float(i) for i in lines[2].split()[1:-1]]

    return potential, NGX, NGY, NGZ, atoms, vectors, coords, atom_types, origin, resolution

def load_obj(name):
#   with open(name, 'rb') as f:
#       return pickle.load(f)
    return pickle.load(open(name, 'rb'))

def get_te(species, num_atoms):
    """ Returns the total number of valence electrons for the unit cell """
    tot_electrons = {'Ru': 8.0, 'Re': 7.0, 'Rf': 261.0, 'Ra': 226.0, 'Rb': 85.47, 'Rn': 8.0,
                    'Rh': 9.0, 'Be': 2.0, 'Ba': 137.33, 'Bh': 264.0, 'Bi': 5.0, 'Bk': 247.0,
                    'Br': 7.0, 'H': 1.0, 'P': 5.0, 'Os': 8.0, 'Es': 252.0, 'Hg': 12.0, 'Ge': 4.0,
                    'Gd': 18.0, 'Ga': 3.0, 'Pr': 13.0, 'Pt': 10.0, 'Pu': 16.0, 'C': 4.0, 'Pb': 4.0,
                    'Pa': 13.0, 'Pd': 10.0, 'Cd': 12.0, 'Po': 6.0, 'Pm': 15.0, 'Hs': 269.0, 'Ho': 21.0,
                    'Hf': 4.0, 'K': 39.1, 'He': 2.0, 'Md': 258.0, 'Mg': 2.0, 'Mo': 6.0, 'Mn': 7.0, 'O': 6.0,
                    'Mt': 268.0, 'S': 6.0, 'W': 6.0, 'Zn': 12.0, 'Eu': 17.0, 'Zr': 91.22, 'Er': 22.0,
                    'Ni': 10.0, 'No': 259.0, 'Na': 1.0, 'Nb': 92.91, 'Nd': 14.0, 'Ne': 8.0, 'Np': 15.0,
                    'Fr': 223.0, 'Fe': 8.0, 'Fm': 257.0, 'B': 3.0, 'F': 7.0, 'Sr': 87.62, 'N': 5.0,
                    'Kr': 8.0, 'Si': 4.0, 'Sn': 4.0, 'Sm': 16.0, 'V': 5.0, 'Sc': 3.0, 'Sb': 5.0,
                    'Sg': 266.0, 'Se': 6.0, 'Co': 9.0, 'Cm': 18.0, 'Cl': 7.0, 'Ca': 40.08, 'Cf': 251.0,
                    'Ce': 12.0, 'Xe': 8.0, 'Lu': 25.0, 'Cs': 132.91, 'Cr': 6.0, 'Cu': 11.0, 'La': 11.0,
                    'Li': 1.0, 'Tl': 3.0, 'Tm': 23.0, 'Lr': 262.0, 'Th': 12.0, 'Ti': 4.0, 'Te': 6.0,
                    'Tb': 19.0, 'Tc': 7.0, 'Ta': 5.0, 'Yb': 24.0, 'Db': 262.0, 'Dy': 20.0, 'I': 7.0,
                    'U': 14.0, 'Y': 88.91, 'Ac': 11.0, 'Ag': 11.0, 'Ir': 9.0, 'Am': 17.0, 'Al': 3.0,
                    'As': 5.0, 'Ar': 8.0, 'Au': 11.0, 'At': 7.0, 'In': 3.0}

    tot_e = 0
    for line, item in enumerate(species):
        tot_e += tot_electrons[str(item)]*num_atoms[line]
    return tot_e

def get_n(species, num_atoms):
    """ Returns 'n' the scaling factor for the PV and SA calcs """
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
    if num_atoms == 'None':
        molar_mass = 0
        for line, item in enumerate(species):
            molar_mass += atomic_mass[str(item)]
        return 1/molar_mass
    else:
        molar_mass = 0
        for line, item in enumerate(species):
            molar_mass += atomic_mass[str(item)]*num_atoms[line]
        return 1/molar_mass

def unit_cell_volume(lattice):
    """ returns the unit cell volume and vector properties """
    def angle(v1, v2):
        angle = np.arccos((np.dot(v1, v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        return angle

    vec = [np.sqrt(lattice[0,0]**2+lattice[0,1]**2+lattice[0,2]**2),
        np.sqrt(lattice[1,0]**2+lattice[1,1]**2+lattice[1,2]**2),
        np.sqrt(lattice[2,0]**2+lattice[2,1]**2+lattice[2,2]**2)]

    cb = angle(lattice[2], lattice[1])
    ca = angle(lattice[2], lattice[0])
    ab = angle(lattice[0], lattice[1])

    vp1 = vec[0]*vec[1]*vec[2]
    vp2 = np.sqrt((1-(np.cos(cb))**2 - (np.cos(ca))**2 - (np.cos(ab))**2) + 2*np.cos(ca)*np.cos(cb)*np.cos(ab))

    vol = vp1 * vp2

    return vol, vec

def process_LOCPOT(file):

    def pot_grid(ng, vasp_pot, lattice, file):
        """ Generate potential grid for all calcs """
        l = 0
        potential_grid = np.zeros(shape=(ng[0], ng[1], ng[2]))
        if file == 'LOCPOT':
            for k in range(ng[2]):
                for j in range(ng[1]):
                    for i in range(ng[0]):
                        potential_grid[i, j, k] = vasp_pot[l]
                        l += 1

        else:
            a = np.sqrt(lattice[0,0]**2+lattice[0,1]**2+lattice[0,2]**2)
            b = np.sqrt(lattice[1,0]**2+lattice[1,1]**2+lattice[1,2]**2)
            c = np.sqrt(lattice[2,0]**2+lattice[2,1]**2+lattice[2,2]**2)

            volume = (a/ng[0]) * (b/ng[1]) * (c/ng[2]) * np.prod(ng)

            for k in range(ng[2]):
                for j in range(ng[1]):
                    for i in range(ng[0]):
                        potential_grid[i,j,k] = vasp_pot[l]/volume
                        l += 1

        return potential_grid

    # read in the file
    with open(file, 'r') as f:
        lines = f.read().split('\n')

    # get the lattice
    lattice = np.zeros(shape=(3,3))
    for i in range(2, 5):
        line = lines[i].split()
        lattice[i-2][0] = float(line[0])
        lattice[i-2][1] = float(line[1])
        lattice[i-2][2] = float(line[2])

    # get the unit cell volume and vectors
    ucv, vec = unit_cell_volume(lattice)

    # get the species and number of each
    species = lines[5].split()
    species[0] = species[0].lstrip().rstrip()
    species = list(filter(None, species))

    num_species = len(species)
    atomcount = lines[6].split()
    num_atoms = []
    total_num_atoms = 0
    for i in atomcount:
        total_num_atoms += mf.c2f(i)
        num_atoms.append(mf.c2f(i))
    print(species)

    # get 'n' the scaling factor for PV and SA calcs
    n = get_n(species, num_atoms)
    print(n)
    # calc crystal density
    cryst_density = (1/n)/(ucv*6.022e23*1e-24)

    # get total number of electrons
    total_num_electrons = get_te(species, num_atoms)

    # get the coordinates for all of the atoms
    coords = np.zeros(shape=(total_num_atoms, 3))
    for i in range(8, total_num_atoms + 8):
        line = lines[i].split()
        coords[i-8, 0] = mf.c2f(line[0])
        coords[i-8, 1] = mf.c2f(line[1])
        coords[i-8, 2] = mf.c2f(line[2])

    # extract NGX, NGY, NGZ as a list
    line = lines[total_num_atoms + 9].split()
    ng = [int(i) for i in line]

    # extract all of the potential information!!!
    k = 0
    bottom = total_num_atoms + 10
    top = bottom + int(np.prod(ng)/5)
    vasp_pot = np.zeros(shape=(int(np.prod(ng))))
    for i in range(bottom, top):
        line = lines[i].split()

        vasp_pot[k] = line[0]
        vasp_pot[k+1] = line[1]
        vasp_pot[k+2] = line[2]
        vasp_pot[k+3] = line[3]
        vasp_pot[k+4] = line[4]

        k += 5

    pg = pot_grid(ng, vasp_pot, lattice, file)

    # return all of the necessary properties
    return lattice, coords, vasp_pot, ng, n, total_num_atoms, total_num_electrons, species, num_atoms, ucv, vec, cryst_density, pg

def process_POSCAR(file):
    with open(file, 'r') as f:
        lines = f.read().split('\n')

    # get the lattice
    lattice = np.zeros(shape=(3,3))
    for i in range(2, 5):
        line = lines[i].split()
        lattice[i-2][0] = float(line[0])
        lattice[i-2][1] = float(line[1])
        lattice[i-2][2] = float(line[2])

    # get the unit cell volume and vectors
    ucv, vec = unit_cell_volume(lattice)

    # get the species and number of each
    species = lines[5].split()
    species[0] = species[0].lstrip().rstrip()
    species = filter(None, species)

    num_species = len(species)
    atomcount = lines[6].split()
    num_atoms = []
    total_num_atoms = 0
    for i in atomcount:
        total_num_atoms += mf.c2f(i)
        num_atoms.append(mf.c2f(i))
    print(species)

    # get 'n' the scaling factor for PV and SA calcs
    n = get_n(species, num_atoms)

    # calc crystal density
    cryst_density = (1/n)/(ucv*6.022e23*1e-24)

    # get total number of electrons
    total_num_electrons = get_te(species, num_atoms)

    # get the coordinates for all of the atoms
    coords = np.zeros(shape=(total_num_atoms, 3))
    for i in range(9, total_num_atoms + 8):
        line = lines[i].split()
        coords[i-9, 0] = mf.c2f(line[0])
        coords[i-9, 1] = mf.c2f(line[1])
        coords[i-9, 2] = mf.c2f(line[2])

    return lattice, coords, n, total_num_atoms, total_num_electrons, species, num_atoms, ucv, vec, cryst_density

def process_CHGCAR(file):

    def pot_grid(ng, vasp_pot, lattice, file):
        """ Generate potential grid for all calcs """
        l = 0
        potential_grid = np.zeros(shape=(ng[0], ng[1], ng[2]))
        if file == 'LOCPOT':
            for k in range(ng[2]):
                for j in range(ng[1]):
                    for i in range(ng[0]):
                        potential_grid[i, j, k] = vasp_pot[l]
                        l += 1

        else:
            a = np.sqrt(lattice[0,0]**2+lattice[0,1]**2+lattice[0,2]**2)
            b = np.sqrt(lattice[1,0]**2+lattice[1,1]**2+lattice[1,2]**2)
            c = np.sqrt(lattice[2,0]**2+lattice[2,1]**2+lattice[2,2]**2)

            volume = (a/ng[0]) * (b/ng[1]) * (c/ng[2]) * np.prod(ng)

            for k in range(ng[2]):
                for j in range(ng[1]):
                    for i in range(ng[0]):
                        potential_grid[i,j,k] = vasp_pot[l]/volume
                        l += 1
            print('GOT INTO THE ELSE STATEMENT')

        return potential_grid

    # read in the file
    with open(file, 'r') as f:
        lines = f.read().split('\n')

    # get the lattice
    lattice = np.zeros(shape=(3,3))
    for i in range(2, 5):
        line = lines[i].split()
        lattice[i-2][0] = float(line[0])
        lattice[i-2][1] = float(line[1])
        lattice[i-2][2] = float(line[2])

    # get the unit cell volume and vectors
    ucv, vec = unit_cell_volume(lattice)

    # get the species and number of each
    species = lines[5].split()
    species[0] = species[0].lstrip().rstrip()
    species = list(filter(None, species))

    num_species = len(species)
    atomcount = lines[6].split()
    num_atoms = []
    total_num_atoms = 0
    for i in atomcount:
        total_num_atoms += mf.c2f(i)
        num_atoms.append(mf.c2f(i))
    print(species)

    # get 'n' the scaling factor for PV and SA calcs
    n = get_n(species, num_atoms)
    print(n)
    # calc crystal density
    cryst_density = (1/n)/(ucv*6.022e23*1e-24)

    # get total number of electrons
    total_num_electrons = get_te(species, num_atoms)

    # get the coordinates for all of the atoms
    coords = np.zeros(shape=(total_num_atoms, 3))
    for i in range(8, total_num_atoms + 8):
        line = lines[i].split()
        coords[i-8, 0] = mf.c2f(line[0])
        coords[i-8, 1] = mf.c2f(line[1])
        coords[i-8, 2] = mf.c2f(line[2])

    # extract NGX, NGY, NGZ as a list
    line = lines[total_num_atoms + 9].split()
    ng = [int(i) for i in line]

    # extract all of the potential information!!!
    k = 0
    bottom = total_num_atoms + 10
    top = bottom + int(np.prod(ng)/5)
    vasp_pot = np.zeros(shape=(int(np.prod(ng))))
    for i in range(bottom, top):
        line = lines[i].split()

        vasp_pot[k] = line[0]
        vasp_pot[k+1] = line[1]
        vasp_pot[k+2] = line[2]
        vasp_pot[k+3] = line[3]
        vasp_pot[k+4] = line[4]

        k += 5

    pg = pot_grid(ng, vasp_pot, lattice, file)

    # return all of the necessary properties
    return lattice, coords, vasp_pot, ng, n, total_num_atoms, total_num_electrons, species, num_atoms, ucv, vec, cryst_density, pg
