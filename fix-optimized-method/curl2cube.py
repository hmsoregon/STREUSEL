"""
Write the curl of the electrostatic potential to a cube file.

"""

import streusel
import glob
import pathlib
import argparse
import numpy as np
from write_cube import write_cube
import matplotlib.pyplot as plt


def _get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("neutral_atom_directory")
    parser.add_argument('-a', '--atom')
    parser.add_argument("output_path")
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
        atom.sample_efield_optimized()
        curl = atom.optimized_method_curl
        efield = atom.efield

        for x in range(75,125):
            plt.imshow(efield[:,:,x], cmap='viridis')
            plt.colorbar()
            plt.savefig(f'efield_{x}.png')

            plt.imshow(curl[:,:,x], cmap='viridis')
            plt.colorbar()
            plt.savefig(f'curl_{x}.png')

        write_cube(cli_args.output_path, curl, atom.coords, atom.atoms.symbols, atom.origin, atom.res)


if __name__ == '__main__':
    main()
