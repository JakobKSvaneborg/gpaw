"""Plugin for ASE so that ASE can read log-file from a GPAW calculation."""
from ase.utils.plugins import ExternalIOFormat


io = ExternalIOFormat(
    desc='GPAW log-file',
    code='+F',  # +=multiple atoms objects, F=accepts a file-descriptor
    module='gpaw.ase_plugin',
    magic='* __  _  _')


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
                        'magnetic_moments'})
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
            cell=dct['unit_cell']['axes'],
            pbc=[p == 'yes' for p in dct['unit_cell']['periodic']],
            initial_magmoms=initial_magmoms)

        kwargs = {}
        if 'energy_contributions' in dct:
            kwargs['energy'] = dct['energy_contributions']['extrapolated']
        if 'forces' in dct:
            kwargs['forces'] = [(x, y, z) for _, x, y, z in dct['forces']]
        if 'stress' in dct:
            kwargs['stress'] = dct['stress']
        # magmoms, dipole
        if kwargs:
            atoms.calc = SinglePointCalculator(atoms, **kwargs)
        images.append(atoms)

    if len(images) == 0:
        raise OSError('Corrupted GPAW-text file!')

    return images
