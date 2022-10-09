import streusel
import glob
import pathlib
import argparse
import numpy as np

def _get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("neutral_atom_directory")
    parser.add_argument('-a', '--atom')
    return parser.parse_args()


def main() -> None:
    cli_args = _get_command_line_arguments()
    print(cli_args.neutral_atom_directory)
    print(cli_args.atom)

    neutral_atom_directory = pathlib.Path(cli_args.neutral_atom_directory)

    if cli_args.atom == None:
        globbed_neutral_atom_directories_to_scan = neutral_atom_directory.joinpath('*/epot_200.cube')
    else:
        globbed_neutral_atom_directories = neutral_atom_directory.joinpath(f'{cli_args.atom}_neut/epot_200.cube')

    for atom_cube_file in glob.glob(str(globbed_neutral_atom_directories)):
        atom = streusel.Molecule(atom_cube_file)
        atom.get_efield()

        atom.sample_efield_original()
        original_surface_area = atom.sarea
        original_volume = atom.original_vol

        atom.sample_efield_optimized()
        optimized_surface_area = atom.sarea
        optimized_volume = atom.optimized_vol

        print(f'original surface area: {original_surface_area}')
        print(f'optimized surface area: {optimized_surface_area}')
        print(f'original volume: {original_volume}')
        print(f'optimized volume: {optimized_volume}')

        # to determine which method is correct, we will use the volume.
        # spheres possess known ratio between surface area and volume.
        # in a vacuum, neutral atoms are spherical.
        # therefore, whichever method returns a surface area that is 3/radius
        # is the correct method

        volume_to_radius_conversion = 3 / (4*np.pi)

        original_radius = np.cbrt(volume_to_radius_conversion * original_volume)
        optimized_radius = np.cbrt(volume_to_radius_conversion * optimized_volume)

        predicted_original_surface_area = 3 / original_radius
        predicted_optimized_surface_area = 3 / optimized_radius

        print(f'original radius from volume of sphere: {original_radius}')
        print(f'optimized radius from volume of sphere: {optimized_radius}')

        print(f'predicted radius original surface area: {predicted_original_surface_area}')
        print(f'predicted radius optimized surface area: {predicted_optimized_surface_area}')


if __name__ == '__main__':
    main()
