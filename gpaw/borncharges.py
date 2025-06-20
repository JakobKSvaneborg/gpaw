import numpy as np
from gpaw.mpi import world
from gpaw.berryphase import polarization_phase, _get_wavefunctions
from ase.parallel import paropen, parprint
from ase.io.jsonio import write_json, read_json
from pathlib import Path


def born_charges_wf(atoms, delta=0.01, cleanup=False,
                    out='born_charges.json', gpw_file=None):

    params = atoms.calc.parameters

    # generate displacement dictionary
    disps_av = _all_disp(atoms, delta)

    # carry out polarization phase calculation
    # for each displacement
    phases_c = {}
    for dlabel in disps_av:
        ia, iv, sign, delta = disps_av[dlabel]
        atoms_d = displace_atom(atoms, ia, iv, sign, delta)
        gpw_wfs = Path(dlabel + '.gpw')
        berryname = Path(dlabel + '_berry-phases.json')
        if not berryname.is_file():
            if not gpw_wfs.is_file():
                gpw_wfs = _get_wavefunctions(atoms_d, params, world,
                                             gpw_wfs, gpw_file=gpw_file)
            # dict with entries phase_c, electronic_phase_c
            # atomic_phase_c, dipole_moment_c
            phase_c = polarization_phase(gpw_wfs, comm=world)

            # only master rank should write
            with paropen(berryname, 'w') as fd:
                write_json(fd, phase_c)

        else:
            # all ranks can read
            with open(berryname, 'r') as fd:
                phase_c = read_json(fd)

        if cleanup:
            if berryname.is_file():
                # remove gpw file
                if world.rank == 0:
                    gpw_wfs.unlink()

        phases_c[dlabel] = phase_c['phase_c']

    results = born_charges(atoms, disps_av, phases_c)
    with paropen(out, 'w') as fd:
        write_json(fd, results)

    return results


def born_charges(atoms, disps_av, phases_c):

    natoms = len(atoms)
    cell_cv = atoms.get_cell()
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    ndisp = len(disps_av)
    parprint('Not using symmetry: ndisp:', ndisp)

    # obtain phi(dr) map
    phi_ascv = np.zeros((natoms, 2, 3, 3), float)
    for dlabel in disps_av:
        ia, iv, sign, delta = disps_av[dlabel]
        isign = [None, 1, 0][sign]
        phi_ascv[ia, isign, :, iv] = phases_c[dlabel]

    # calculate dphi / dr
    # exploit +- displacement
    dphi_acv = phi_ascv[:, 1] - phi_ascv[:, 0]
    # mod 2 pi
    mod_acv = np.round(dphi_acv / (2 * np.pi)) * 2 * np.pi
    dphi_acv -= mod_acv
    # transform to cartesian
    dphi_avv = np.array([np.dot(dphi_cv.T, cell_cv).T for dphi_cv in dphi_acv])
    dphi_dr_avv = dphi_avv / (2.0 * delta)

    # calculate polarization change and born charges
    dP_dr_avv = dphi_dr_avv / (2 * np.pi * vol)
    Z_avv = dP_dr_avv * vol

    # check acoustic sum rule: sum_a Z_aij = 0 for all i,j
    asr_vv = np.sum(Z_avv, axis=0)
    asr_dev = np.abs(asr_vv).max() / natoms
    assert asr_dev < 1e-1, f'Acoustic sum rule violated: {asr_vv}'

    # correct to match acoustic sum rule
    Z_avv -= asr_vv[None, :, :] / natoms

    results = {'Z_avv': Z_avv, 'sym_a': sym_a}

    return results


def _cartesian_label(ia, iv, sign):
    """Generate name from (ia, iv, sign).
    ia ... atomic_index
    iv ... cartesian_index
    sign ... +-
    """

    sym_v = 'xyz'[iv]
    sym_s = ' +-'[sign]
    return f'{ia}{sym_v}{sym_s}'


def _all_avs(atoms):
    """Generate ia, iv, sign for all displacements."""
    for ia in range(len(atoms)):
        for iv in range(3):
            for sign in [-1, 1]:
                yield (ia, iv, sign)


def _all_disp(atoms, delta):
    all_disp = {}
    for dd, avs in enumerate(_all_avs(atoms)):
        dd = int(dd)
        lavs = _cartesian_label(*avs)
        label = f'disp_{dd:03d}_' + lavs
        all_disp[label] = (*avs, delta)
    return all_disp


def displace_atom(atoms, ia, iv, sign, delta):
    new_atoms = atoms.copy()
    pos_av = new_atoms.get_positions()
    pos_av[ia, iv] += sign * delta
    new_atoms.set_positions(pos_av)
    return new_atoms
