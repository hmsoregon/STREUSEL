import streusel
import glob
import pytest

"""
def test_gaussian_implementations_for_volumes() -> None:
    # arrange -> arrange the data you are testing
    neutral_atom_cube_files = '/home/he/bin/hba_submitted_data/ionic_radii/neutral_atom_data/*/epot_200.cube'

    # act -> calculate the things you want to test
    for atom_cube in glob.glob(neutral_atom_cube_files):
        mol = streusel.Molecule(atom_cube)
        mol.get_efield()
        mol.sample_efield_original()
        original_method_volume = mol.vol

        mol.sample_efield_optimized()
        optimized_method_volume = mol.vol

        # assert -> determine if the test passed or not
        assert original_method_volume == pytest.approx(optimized_method_volume)
"""


def test_gaussian_implementations_for_surface_areas() -> None:
    # arrange -> arrange the data you are testing
    neutral_atom_cube_files = '/home/he/bin/hba_submitted_data/ionic_radii/neutral_atom_data/*/epot_200.cube'

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
