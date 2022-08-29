from . import geometry_manipulation_functions as gm
from . import process_functions as pf
from . import math_functions as mf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import shutil
import math
import os
import scipy.ndimage
import itertools


class Material:
    """ Class for properties of material """
    def __init__(self, file):
        lattice, coords, vasp_pot, ng, n, total_num_atoms, total_num_electrons, species, num_atoms, ucv, vec, cryst_density, pg = pf.process_LOCPOT(file)

        # assign material properties
        self.type = file
        self.lattice = lattice
        self.coords = coords
        self.ngs = ng
        self.vasp_pot = vasp_pot
        self.n = n
        self.species = species
        self.num_atoms = num_atoms
        self.total_num_atoms = total_num_atoms
        self.ucv = ucv
        self.vecs = vec
        self.cryst_density = cryst_density
        self.pot_grid = pg
        self.coords_and_atoms = gm.combine_coordsANDatoms(coords, species, num_atoms=num_atoms)

    def save_obj(self, name_path):
        """ save material as a dictionary output """
        with open(name_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def get_efield(self):
        n = self.n
        ng = self.ngs
        vecs = self.vecs

        pgrid = self.pot_grid
        res = [vecs[0]/ng[0], vecs[1]/ng[1], vecs[2]/ng[2]]

        grad_x, grad_y, grad_z = np.gradient(pgrid[:,:,:], res[0], res[1], res[2])

        grad_mag = mf.grad_magnitude(grad_x, grad_y, grad_z)

        self.efield = grad_mag

        # oefield, oefield_var = ef.get_field(grad_mag, ng, res)
        # self.oefield = oefield

    def sample_field(self):
        cv = 1e-5
        # cv = 1e-6
        efield = self.efield
        ngs = self.ngs
        vecs = self.vecs

        vacuum = []
        non_vacuum = []
        surface = []
        vac_ijk = []
        nvac_ijk = []
        surf_ijk = []

        v = 0
        n = 0
        sarea = 0
        mi = (vecs[1]/ngs[1]) * (vecs[2]/ngs[2])
        mj = (vecs[0]/ngs[0]) * (vecs[2]/ngs[2])
        mk = (vecs[0]/ngs[0]) * (vecs[1]/ngs[1])
        print(mi, mj, mk)
        efield_view = efield[:-1, :-1, :-1]
        efield_diff = efield_view - efield[1:, 1:, 1:]
        np.absolute(efield_diff, out=efield_diff)
        is_vacuum = efield_diff < cv
        is_non_vacuum = np.logical_not(is_vacuum)
        vacuum = efield_view[is_vacuum]
        non_vacuum = efield_view[is_non_vacuum]
        nvac_ijk = np.argwhere(is_non_vacuum)
        weights = np.zeros((3, 3, 3))

        weights[1, 1, 2] = 1
        weights[1, 1, 0] = 1
        weights[:, :, 1] = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]

        convolution_result = scipy.ndimage.convolve(
            input=is_vacuum.view(np.int8),
            weights=weights,
            mode='wrap',
        )
        np.multiply(
            convolution_result,
            is_non_vacuum.view(np.int8),
            out=convolution_result,
        )
        surface_mask = convolution_result > 0
        surf_ijk = np.argwhere(surface_mask)
        sarea = np.sum(surface_mask) * mk
        sarea2 = np.sum(convolution_result) * mk
        print('surface areas', sarea, sarea2)

        """
        for i in tqdm(range(ngs[0] - 1)):
            for j in range(ngs[1] - 1):
                for k in range(ngs[2] - 1):
                    if np.absolute(efield[i,j,k] - efield[i+1, j+1, k+1]) <= cv:
                        # if efield[i,j,k] - efield[i + 1, j + 1, k + 1] <= cv:
                        vacuum.append(efield[i,j,k])
                        v += 1
                    else:
                        non_vacuum.append(efield[i,j,k])
                        nvac_ijk.append([i, j, k])
                        n += 1
                    if (n+v) == 1:
                        surface.append(efield[i,j,k])
                        surf_ijk.append([i, j, k])
                        sarea += mk
                    elif v == 2 or n == 2:
                        v = 0
                        n = 0
        """
        """
        for i in tqdm(range(ngs[0]-2, ngs[0]-1)):
            for j in range(ngs[1]-2, ngs[1]-1):
                for k in range(ngs[2]-2, ngs[2]-1):
                    if np.absolute(efield[i,j,k]-efield[i+1,j+1,k+1]) <=cv:
                        vacuum.append(efield[i,j,k])
                        v+=1
                    else:
                        non_vacuum.append(efield[i,j,k])
                        nvac_ijk.append([i,j,k])
                        n+=1
                    if (n+v) == 1:
                        surface.append(efield[i,j,k])
                        surf_ijk.append([i,j,k])
                    elif v == 2 or n == 2:
                        v = 0
                        n = 0
        v = 0
        n = 0
        for j in tqdm(range(ngs[1]-1)):
            for k in range(ngs[2]-1):
                for i in range(ngs[0]-1):
                    if np.absolute(efield[i,j,k] - efield[i+1,j+1,k+1]) <= cv:
                        #vacuum.append(efield[i,j,k])
                        v+=1
                    else:
                        #non_vacuum.append(efield[i,j,k])
                        #nvac_ijk.append([i,j,k])
                        n+=1
                    # commenting out .appends in this one, because don't want duplicate pts
                    if (n+v) == 1:
                        #surface.append(efield[i,j,k])
                        sarea += mi
                    elif v==2 or n==2:
                        v = 0
                        n = 0
        v = 0
        n = 0
        for k in tqdm(range(ngs[2]-1)):
            for i in range(ngs[0]-1):
                for j in range(ngs[1]-1):
                    if np.absolute(efield[i,j,k]-efield[i+1,j+1,k+1]) <= cv:
                        v+=1
                    else:
                        n += 1
                    if (n+v) == 1:
                        sarea += mj
                    elif v == 2 or n == 2:
                        v = 0
                        n = 0
        """
        self.vac = vacuum
        self.nvac = non_vacuum
        self.surf = surface
        self.sijk = surf_ijk
        self.nijk = nvac_ijk
        self.surf_area = sarea * float(1e-20)*float(self.n)*float(6.022e23)


    def calc_volume(self, vol_type):
        ng = self.ngs
        vecs = self.vecs
        vol_per_cube = (vecs[0]/ng[0]) * (vecs[1]/ng[1]) * (vecs[2]/ng[2])

        if vol_type == 'MOF_pore':
            vol = float(vol_per_cube)*len(self.vac)*float(1e-24)*float(self.n)*float(6.022e23)
        elif vol_type == 'MOF_pore_orig':
            vol = float(vol_per_cube)*len(self.vac)*float(1e-24)*float(self.n)*float(6.022e23) * 8
        elif vol_type == 'molecule':
            vol = float(vol_per_cube) * len(self.nvac)

        self.vol = vol

    def calc_surfacearea(self):
        ng = self.ngs
        vecs = self.vecs
        sijk = np.array(self.sijk[::10])
        print(sijk.shape)
        nvac = np.array(self.nijk[::10])
        print(nvac.shape)
        # vx = [nvac[r][0]*ng[0] for r in range(vr)]
        # vy = [nvac[r][1]*ng[1] for r in range(vr)]
        # vz = [nvac[r][2]*ng[2] for r in range(vr)]

        # try:
        # cvf.create_vacs(vx, vy, vz)
        # except Exception:
        # cvf.remove_vacs(vx)
        path = '/home/he/bin/github_clones/STREUSEL/examples/'
        try:
            os.makedirs(path + '/sepvac/')
        except OSError:
            shutil.rmtree(path + '/sepvac/')
            os.makedirs(path + '/sepvac/')
        ca = np.zeros(shape=(6,4))
        surface_area = 0

        mi = (vecs[1]/ng[1]) * (vecs[2]/ng[2])
        mj = (vecs[0]/ng[0]) * (vecs[2]/ng[2])
        mk = (vecs[0]/ng[0]) * (vecs[1]/ng[1])

        # avgms = (mi+mj+mk)/3
        avgms = (mi*mj + mi*mk + mk*mj)
        avgsurf = float(avgms) * len(self.surf) * float(1e-20)*float(self.n)*float(6.022e23)
        print(avgms)
        print(avgsurf)

        hik = np.sqrt(mi**2 + mk**2)
        hij = np.sqrt(mi**2 + mj**2)
        hkj = np.sqrt(mk**2 + mj**2)
        p = (hik + hij + hkj)/3
        planearea = np.sqrt(p*(p-hik)*(p-hij)*(p-hkj))
        planesurf = float(planearea * avgms)*len(self.surf)*float(1e-20)*float(self.n)*float(6.022e23)
        print(planesurf)
        print(ng[0]*ng[1]*ng[2])
        print(len(self.vac) + len(self.nvac))
        print(len(self.surf))
        """
        print(nvac[:].shape)
        cvf.create_vacs(nvac[:,0], nvac[:,1], nvac[:,2])

        tic = time.time()
        print('pre sa loop >>')
        x = sijk[:,0]
        y = sijk[:,1]
        z = sijk[:,2]
        s = 0
        print(len(x), len(y), len(z))
        for ii in tqdm(range(len(x) - 1)):
            ca[0] = [x[ii]+1, y[ii], z[ii], mi]
            ca[1] = [x[ii]-1, y[ii], z[ii], mi]
            ca[2] = [x[ii], y[ii]+1, z[ii], mj]
            ca[3] = [x[ii], y[ii]-1, z[ii], mj]
            ca[4] = [x[ii], y[ii], z[ii]+1, mk]
            ca[5] = [x[ii], y[ii], z[ii]-1, mk]
            print(s)
            s += float(cvf.mpi_calc(ca))
        """
        sarea_total = float(s) * float(self.n) * float(1e-20) * float(6.022e23)

        print(sarea_total)
        self.surf_area = sarea_total


    def original_sampling_method(self):
        """ Samples electric field generated by get_efield(self) """
        """ Returns vacuum, non_vacuum, surface, and surface i,j,k values """
        cv = 1e-5 # sigfigs

        ng = self.ngs
        potential_grid = self.efield
        vecs = self.vecs

        cube = [2,2,2]
        travelled = [0,0,0]
        origin = [0,0,0]
        # try:
        vacuum = []
        non_vacuum = []
        surface = []

        vacpot = []
        nvacpot = []
        surfpot = []
        surf_ijk = []
        cubepot = []

        n = 0
        v = 0

        for i in tqdm(range(0, ng[0], cube[0])):
            for j in range(0, ng[1], cube[1]):
                for k in range(0, ng[2], cube[2]):
                    origin = [float(i)/ng[0], float(j)/ng[1], float(k)/ng[2]]
                    n_origin = np.zeros(shape = (3))
                    n_origin[0] = int(i)
                    n_origin[1] = int(j)
                    n_origin[2] = int(k)

                    density_cube = np.zeros(shape=(cube[0], cube[1], cube[2]))

                    for x in range(0, cube[0]):
                        for y in range(0, cube[1]):
                            for z in range(0, cube[2]):
                                xv = n_origin[0]+travelled[0]+x
                                yv = n_origin[1]+travelled[1]+y
                                zv = n_origin[2]+travelled[2]+z

                                zv = zv - ng[2]*round(zv/ng[2])
                                yv = yv - ng[1]*round(yv/ng[1])
                                xv = xv - ng[0]*round(xv/ng[0])

                                density_cube[x, y, z] = potential_grid[int(xv), int(yv), int(zv)]
                    cube_potential = density_cube.mean()
                    cubepot.append(cube_potential)
                    cube_var = np.var(density_cube)
                    if cube_var <= cv:
                        vacuum.append(origin)
                        vacpot.append(cube_potential)
                        v += 1
                    else:
                        non_vacuum.append(origin)
                        nvacpot.append(cube_potential)
                        n += 1
                    if (n+v) == 2:
                        surface.append(origin)
                        surfpot.append(cube_potential)
                        surf_ijk.append([[i, j, k]])
                    elif v == 2 or n == 2:
                        v = 0
                        n = 0

        # vol_per_cube = (vecs[0]/ng[0]) * (vecs[1]/ng[1]) * (vecs[2]/ng[2])

        # mof_vol = 8*float(vol_per_cube)*len(vacuum)*float(1e-24)*float(self.n)*float(6.022e23)
        # ubound_vol = vol_per_cube * len(non_vacuum) * 8
        self.vac = vacuum
        self.nvac = non_vacuum
        self.surf = surface
        self.sijk = surf_ijk
        # self.ovol = ubound_vol
