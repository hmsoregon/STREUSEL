import streusel
import glob
import pytest
import math
import numpy as np
import matplotlib.pyplot as plt
from tests.write_cube import write_cube

def generate_sphere(radius=1, samples=10000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_ = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius_
        z = math.sin(theta) * radius_

        points.append((x, y, z))
    sphere_array = np.array(points)*radius
    voxel_positions = np.linspace(-radius*1.1, radius*1.1, 200*radius)
    x_index_voxels = np.digitize(sphere_array[:, 0], voxel_positions)
    y_index_voxels = np.digitize(sphere_array[:, 1], voxel_positions)
    z_index_voxels = np.digitize(sphere_array[:, 2], voxel_positions)

    return x_index_voxels, y_index_voxels, z_index_voxels


def test_optimized_surface_area_clean_sphere() -> None:
    radius = 2
    # arrange -> arrange the data you are testing
    x_voxels, y_voxels, z_voxels = generate_sphere(radius=radius)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_voxels, y_voxels, z_voxels)
    plt.savefig("clean_sphere.png")
    sphere_indices = np.vstack((x_voxels, y_voxels, z_voxels)).T
    sphere_centroid = [int(cent) for cent in np.mean(sphere_indices, axis=0)]

    entire_tensor = np.zeros(shape=(x_voxels.max()+10,y_voxels.max()+10,z_voxels.max()+10))
    # set indices of sphere to 1
    entire_tensor[sphere_indices[:,0], sphere_indices[:,1], sphere_indices[:,2]] = 1e-5
    write_cube(
        filepath="clean_sphere.cube",
        vox_grid=entire_tensor,
        coords=np.array([[0,0,0]]),
        elements=[str(1)],
        origin=[0,0,0],
        res=(0.2*radius, 0.2*radius, 0.2*radius),
    )
    sphere = streusel.Molecule("clean_sphere.cube")

    print("CLEAN SPHERE OPTIMIZED")
    sphere.get_efield()
    sphere.sample_efield_optimized()
    print(sphere.sarea)
    print(sphere.optimized_vol)

    vol_r = np.power((3*sphere.optimized_vol)/(4*np.pi), (1/3))
    sa_r = np.power((sphere.sarea)/(4*np.pi), 0.5)
    print(vol_r)
    print(sa_r)
    
    print("CLEAN SPHERE ORIGINAL")
    sphere.sample_efield_original()
    print(sphere.sarea)
    print(sphere.original_vol)

    vol_r = np.power((3*sphere.original_vol)/(4*np.pi), (1/3))
    sa_r = np.power((sphere.sarea)/(4*np.pi), 0.5)
    print(vol_r)
    print(sa_r)

    return
    
    sphere.sample_efield_original()
    print(sphere.sarea)
    print(sphere.original_vol)

    vol_r = np.power((3*sphere.original_vol)/(4*np.pi), (1/3))
    sa_r = np.power((sphere.sarea)/(4*np.pi), 0.5)
    print(vol_r)
    print(sa_r)

    return

    # act -> calculate the surface area
    for atom_cube in glob.glob(neutral_atom_cube_files):
        mol = streusel.Molecule(atom_cube)
        mol.get_efield()
        mol.sample_efield_original()
        original_method_surface_area = mol.sarea

        mol.sample_efield_optimized()
        optimized_method_surface_area = mol.sarea

        # assert -> determine if the test is passed
        assert original_method_surface_area == pytest.approx(optimized_method_surface_area)


def test_optimized_surface_area_wonky_sphere() -> None:

    # arrange -> arrange the data you are testing
    radius = 2
    # generate clean sphere
    large_x_voxels, large_y_voxels, large_z_voxels = generate_sphere(radius=radius)
    # generate smaller sphere
    small_x_voxels, small_y_voxels, small_z_voxels = generate_sphere(radius=1)

    large_sphere_indices = np.vstack((large_x_voxels, large_y_voxels, large_z_voxels)).T
    large_sphere_centroid = [int(cent) for cent in np.mean(large_sphere_indices, axis=0)]

    small_sphere_indices = np.vstack((small_x_voxels, small_y_voxels, small_z_voxels)).T
    small_sphere_centroid = [int(cent) for cent in np.mean(small_sphere_indices, axis=0)]

    translated_sm_sphere_indices = np.copy(small_sphere_indices)
    translated_sm_sphere_indices[:,0] += (large_sphere_centroid[0] - small_sphere_centroid[0])
    translated_sm_sphere_indices[:,1] += (large_sphere_centroid[1] - small_sphere_centroid[1])
    translated_sm_sphere_indices[:,2] += (large_sphere_centroid[2] - small_sphere_centroid[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(large_x_voxels, large_y_voxels, large_z_voxels)
    ax.scatter(translated_sm_sphere_indices[:,0], translated_sm_sphere_indices[:,1], translated_sm_sphere_indices[:,2])
    plt.savefig("wonky_spheres.png")

    entire_tensor = np.zeros(shape=(large_x_voxels.max()+10,large_y_voxels.max()+10,large_z_voxels.max()+10))
    # set indices of sphere to 1
    entire_tensor[large_sphere_indices[:,0], large_sphere_indices[:,1], large_sphere_indices[:,2]] = 1e-5
    # add in center sphere
    entire_tensor[translated_sm_sphere_indices[:,0], translated_sm_sphere_indices[:,1], translated_sm_sphere_indices[:,2]] = 1e-5

    write_cube(
        filepath="wonky_spheres.cube",
        vox_grid=entire_tensor,
        coords=np.array([[0,0,0]]),
        elements=[str(1)],
        origin=[0,0,0],
        res=(0.2*radius, 0.2*radius, 0.2*radius),
    )
    sphere = streusel.Molecule("wonky_spheres.cube")

    sphere.get_efield()
    sphere.sample_efield_optimized()
    print("WONKY SPHERE OPTIMIZED")
    print(sphere.sarea)
    print(sphere.optimized_vol)

    vol_r = np.power((3*sphere.optimized_vol)/(4*np.pi), (1/3))
    sa_r = np.power((sphere.sarea)/(4*np.pi), 0.5)
    print(vol_r)
    print(sa_r)
    print("WONKY SPHERE ORIGINAL")
    sphere.sample_efield_original()
    print(sphere.sarea)
    print(sphere.original_vol)

    vol_r = np.power((3*sphere.original_vol)/(4*np.pi), (1/3))
    sa_r = np.power((sphere.sarea)/(4*np.pi), 0.5)
    print(vol_r)
    print(sa_r)


    return

