"""Plugin for ASE so that ASE can read log-files from GPAW calculations."""
from ase.utils.plugins import ExternalIOFormat


io = ExternalIOFormat(
    desc='GPAW log-file',
    code='+F',  # +=multiple atoms objects, F=accepts a file-descriptor
    module='gpaw.ase_plugin',
    magic=b'* __  _  _')


def read_gpaw_log(filedesc, index):
    from ase import Atoms
    from gpaw.io.log_file_reader import parse
    from ase.calculators.singlepoint import SinglePointCalculator
    dicts = parse(filedesc,
                  keys={'atoms',
                        'unit_cell',
                        'energy_contributions',
                        'forces',
                        'stress',
                        'magnetic_moments',
                        'dipole'})
    images = []
    for dct in dicts[index]:
        symbols = []
        positions = []
        initial_magmoms = []
        for _, symbol, x, y, z, (_, _, m) in dct['atoms']:
            symbols.append(symbol)
            positions.append((x, y, z))
            initial_magmoms.append(m)

        atoms = Atoms(
            symbols,
            positions,
            cell=[axis for _, *axis, _, _ in dct['unit_cell']],
            pbc=[pbc == 'yes' for pbc, *_ in dct['unit_cell']],
            magmoms=initial_magmoms)

        kwargs = {}
        if 'energy_contributions' in dct:
            kwargs['energy'] = dct['energy_contributions']['extrapolated']
        if 'forces' in dct:
            kwargs['forces'] = [(x, y, z) for _, x, y, z in dct['forces']]
        if 'stress' in dct:
            kwargs['stress'] = dct['stress']
        if 'magnetic_moments' in dct:
            kwargs['magmoms'] = [
                (x, y, z) for _, x, y, z in dct['magnetic_moments']]
        if 'dipole' in dct:
            kwargs['dipole'] = dct['dipole']
        if kwargs:
            atoms.calc = SinglePointCalculator(atoms, **kwargs)
        images.append(atoms)

    if len(images) == 0:
        raise OSError('Corrupted GPAW log-file!')

    return images
