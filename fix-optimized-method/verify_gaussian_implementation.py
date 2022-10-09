import streusel
import glob
import pytest
import pathlib
import argparse

def _get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("neutral_atom_directory", type=str)
    parser.add_argument('-a', '--atom', type=str)
    return parser.parse_args()


def main() -> None:
    cli_args = _get_command_line_arguments()
    neutral_atoms_directory = cli_args.neutral_atom_directory

    if cli_args.atom == 'H':
        print('hello')


if __name__ == '__main__':
    main()



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
