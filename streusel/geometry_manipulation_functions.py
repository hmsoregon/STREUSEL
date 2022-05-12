from . import math_functions as mf 
import pandas as pd
import numpy as np
import pickle
import math
import os

def gen_speciesANDatoms_line(ncoords):
        species = []
        
        for index, row in ncoords.iterrows():
            species.append(row['atom'])
        atoms = [len(list(group)) for key, group in groupby(species)]
        fspecies = list(set(species))
        return fspecies, atoms

def combine_coordsANDatoms(coords, species, num_atoms=None):
    # Pairs each coordinate with an atom type
	if num_atoms != None:
		species_num = []
		for index, atom in enumerate(species):
			# print(atom, num_atoms[index])
			for i in range(num_atoms[index]):
				species_num.append(atom)
		coordsDF = pd.DataFrame(data=coords, columns=['x','y','z'])
		# print(coordsDF)
		atomsDF = pd.DataFrame(data=species_num, columns=['atom'])
		totalDF = pd.concat([coordsDF, atomsDF], axis=1)
	else:
		coordsDF = pd.DataFrame(data=coords, columns=['x', 'y', 'z'])
		atomsDF = pd.DataFrame(data=species, columns=['atom'])
		totalDF = pd.concat([coordsDF, atomsDF], axis=1)
	return totalDF

def graph3D_DataFrame(b, h=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(b['x'].astype('float'), b['y'].astype('float'), b['z'].astype('float'))
    try:
        ax.scatter(h['x'].astype('float'), h['y'].astype('float'), h['z'].astype('float'), c='red')
    except Exception:
        pass
    plt.show()

def distance(point1, point2, data_type=None):
	if data_type == 'dataframe':
		return math.sqrt((point1['x']-point2['x'])**2 + (point1['y']-point2['y'])**2 + (point1['z']-point2['z'])**2)
	else:
		return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)
