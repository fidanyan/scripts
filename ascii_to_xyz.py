#! /usr/bin/env python
""" This script takes Phonopy normal modes in ascii format
    and returns animation as a XYZ file and a set of AIMS geometry files.
    I took ascii-related functions from
    'ascii-phonons/addons/vsim2blender/ascii_importer.py',
    but got rid of Blender's mathutils (replaced them by numpy operations).
    A user should specify:
        - input file in Phonopy .ascii format
        - mode index
        - supercell matrix
        - amplitude of maximal atomic displacements
"""

import os
import sys
import numpy as np
import re
from collections import namedtuple
from ase import Atoms
from ase.io import write


n_frames = 20   # The number of frames for an animation

np.set_printoptions(precision=8, suppress=True, linewidth=120)


class Mode(namedtuple('Mode', 'freq qpt vectors')):
    """
    Collection of vibrational mode data imported from a v_sim ascii file

    :param freq: Vibrational frequency
    :type freq: float
    :param qpt: **q**-point of mode
    :type qpt: 3-list of reciprocal space coordinates
    :param vectors: Eigenvectors
    :type vectors: Nested list; 3-lists of complex num-s corresponding to atoms
    """
    pass


def _check_if_reduced(filename):
    """
    Scan a .ascii file for the "reduced" keyword

    :params filename: v_sim .ascii file to check
    :type filename: float

    :returns: Boolean
    """
    with open(filename, 'r') as f:
        f.readline()  # Skip header
        for line in f:
            if 'reduced' in line:
                return True
        else:
            return False


def import_vsim(filename):
    """
    Copied from ascii-phonons/addons/vsim2blender/ascii_importer.py

    Import data from v_sim ascii file,
    including lattice vectors, atomic positions and phonon modes

    :param filename: Path to .ascii file

    :returns: cell_vsim, positions, symbols, vibs

    :return cell_vsim: Lattice vectors in v_sim format
    :rtype: 2x3 nested lists of floats
    :return positions: Atomic positions
    :rtype: list of 3-Vectors
    :return symbols:   Symbols corresponding to atomic positions
    :rtype: list of strings
    :return vibs:      Vibrations
    :rtype: list of "Mode" namedtuples

    """
    with open(filename, 'r') as f:
        f.readline()  # Skip header
        # Read in lattice vectors (2-row format) and cast as floats
        cell_vsim = [[float(x) for x in f.readline().split()],
                     [float(x) for x in f.readline().split()]]
        # Read in all remaining non-commented lines as positions/symbols,
        # and commented lines to a new array
        positions, symbols, commentlines = [], [], []
        for line in f:
            if line[0] != '#' and line[0] != '\!':
                line = line.split()
                position = [float(x) for x in line[0:3]]
                symbol = line[3]
                positions.append(np.asarray(position))
                symbols.append(symbol)
            else:
                commentlines.append(line.strip())

    # remove comment characters and implement linebreaks
    # (linebreak character \)

    for index, line in enumerate(commentlines):
        while line[-1] == '\\':
            line = line[:-1] + commentlines.pop(index+1)[1:]
        commentlines[index] = line[1:]

    # Import data from commentlines
    vibs = []
    for line in commentlines:
        vector_txt = re.search('qpt=\[(.+)\]', line)
        if vector_txt:
            mode_data = vector_txt.group(1).split(';')
            qpt = [float(x) for x in mode_data[0:3]]
            freq = float(mode_data[3])
            vector_list = [float(x) for x in mode_data[4:]]
            vector_set = [vector_list[6*i:6*i+6]
                          for i in range(len(positions))]
            complex_vectors = [[complex(x[0], x[3]),
                                complex(x[1], x[4]),
                                complex(x[2], x[5])] for x in vector_set]
            vibs.append(Mode(freq, qpt, complex_vectors))

    if _check_if_reduced(filename):
        print("Reduced coordinates are not expected.")
        sys.exit()
        # positions = _reduced_to_cartesian(positions, cell_vsim)

    return (cell_vsim, positions, symbols, vibs)


def cell_vsim_to_vectors(cell_vsim):
    """
    Copied from ascii-phonons/addons/vsim2blender/ascii_importer.py

    Convert between v_sim 6-value lattice vector format
    (`ref <http://inac.cea.fr/L_Sim/V_Sim/sample.html>`)
    and set of three Cartesian vectors

    :param cell_vsim: Lattice vectors in v_sim format
    :type cell_vsim: 2x3 nested lists

    :returns: Cartesian lattice vectors
    :rtype: 3-list of 3-Vectors
    """
    dxx, dyx, dyy = cell_vsim[0]
    dzx, dzy, dzz = cell_vsim[1]
    return np.asarray([[dxx, 0., 0.], [dyx, dyy, 0.], [dzx, dzy, dzz]])


# ========================= __main__ =========================
if __name__ == "__main__":
    if len(sys.argv) == 7:
        inname = sys.argv[1]
        mode_index = int(sys.argv[2])
        supercell_dims = [int(i) for i in sys.argv[3:6]]
        amp = float(sys.argv[6])
    else:
        print("ERROR: 6 arguments are needed:\n"
              "\tinname\n"
              "\tmode_index\n"
              "\tNx Ny Nz - supercell dimensions\n"
              "\tamp - amplitude of the maximal atomic displacement")
        sys.exit(-1)

    # Reading a file using modified ascii-phonons routines:
    cell_vsim, positions, symbols, vibs = import_vsim(inname)

    cell = cell_vsim_to_vectors(cell_vsim)
    supercell = np.matmul(cell, np.diag(supercell_dims))
    recip_cell = np.linalg.inv(cell) * 2*np.pi
    positions = np.asarray(positions)
    print("cell:")
    print(cell)
    print("recip_cell:")
    print(recip_cell)
    print("supercell coefficients:")
    print(supercell_dims)
    print("supercell:")
    print(supercell)
    print("symbols:")
    print(symbols)
    print("positions (unitcell):")
    for p in positions:
        print(p)
    print("Selected mode:")
    print(vibs[mode_index])

    # Filling a supercell:
    pos_supercell = []
    symbols_supercell = []
    for l in range(supercell_dims[0]):
        for m in range(supercell_dims[1]):
            for n in range(supercell_dims[2]):
                for a, pos in enumerate(positions):
                    # add all lattice vectors with the corresponding supercell
                    # coefficients to the unit cell positions:
                    pos_supercell.append(pos
                                         + np.dot(cell,
                                                  np.asarray([l, m, n])))
                    symbols_supercell.append(symbols[a])

    pos_supercell = np.asarray(pos_supercell)
    print("supercell:\t%i atoms" % len(pos_supercell))

    # Reading and normalizing the displacement vector for the given mode_index:
    disp = np.asarray(vibs[mode_index].vectors)
    print("disp:")
    print(disp)
    # normalizes all lines (vectors) in 2D array:
    max_disp = np.max(np.linalg.norm(disp, axis=1))
    if max_disp == 0:
        print("Error: the maximal atomic displacement is 0. "
              "Something is wrong, aborting.")
        sys.exit(-1)
    norm_disp = disp / max_disp * amp
    print("norm_disp:")
    print(norm_disp)

    pos_new = np.copy(pos_supercell)
    qpt_cartesian = np.dot(recip_cell, vibs[mode_index].qpt)
    print("qpt_cartesian:")
    print(qpt_cartesian)

    outname_xyz = "vib." + inname + "mode%i.xyz" % mode_index
    if os.path.exists(outname_xyz):
        os.remove(outname_xyz)
    for frame in range(n_frames):
        for a, r in enumerate(pos_supercell):
            exponent = np.exp(1.j * (np.dot(r, qpt_cartesian)
                                     - 2*np.pi * frame/n_frames))
            pos_new[a] = r + (norm_disp[a % len(symbols)] * exponent).real
        # pos_new = pos_new.real
        max_disp = np.max(np.linalg.norm(pos_new - pos_supercell, axis=1))
        print("Frame %i/%i, max displacement is %f"
              % (frame, n_frames, max_disp))

        atoms = Atoms(symbols=symbols_supercell,
                      positions=pos_new,
                      cell=supercell,
                      pbc=True)

        outname = "geometry.%s.mode%i.amp%.3f.frame%03d.in" % (inname,
                                                               mode_index,
                                                               amp,
                                                               frame)
        info = ("Generated from the file '%s',\n"
                "# phonon mode %i (counted from 0), "
                "freq=%.4f cm^-1 (should be, "
                "but check Phonopy outputs to be sure),\n"
                "# frame %i/%i,\n"
                "# maximal atomic displacement is %.6f"
                % (inname,
                   mode_index,
                   vibs[mode_index].freq,
                   frame,
                   n_frames,
                   max_disp)
                )
        print("Saving frame %i to '%s'" % (frame, outname))
        write(outname, atoms, info_str=info,  append=False, format='aims')
        write(outname_xyz,
              atoms,
              append=True,
              comment="frame %i/%i" % (frame, n_frames),
              format='xyz')

    print("Done.")
    sys.exit(0)
