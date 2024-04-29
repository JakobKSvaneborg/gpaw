import pytest
import numpy as np
from gpaw.mpi import world
from gpaw.atom.generator2 import generate


@pytest.mark.hybrids
@pytest.mark.skipif(world.size > 1, reason='Not parallelized')
def test_diamond(in_tmp_dir, add_cwd_to_setup_paths):

    from gpaw.atom.generator import Generator
    from gpaw.atom.configurations import parameters
    from gpaw.atom.basis import BasisMaker

    args = parameters['C']
    generator = Generator('C', 'PBE', configuration='1s2')
    generator.N *= 2
    generator.run(write_xml=True, exx=True, **args)
    bm = BasisMaker(generator, run=False)
    basis = bm.generate(zetacount=4, polarizationcount=2,
                        energysplit=0.0001, rcutmax=12.0)
    basis.write_xml()

    parameters = {'symbol': 'C', 'Z': 6, 'xc': 'PBE',
                  'projectors': '2s,s,2p,p,d', 'radii': [1.2],
                  'scalar_relativistic': False, 'r0': 1.2,
                  'v0': None,
                  'nderiv0': 2, 'pseudize': ('poly', 4),
                  'omega': 0.11}

    gen = generate(**parameters)
    setup = gen.make_paw_setup()
    setup.write_xml()

    from gpaw import GPAW
    from ase import Atoms
    from ase.build import bulk
    from ase.units import Hartree

    atoms = Atoms('C')
    atoms.center(vacuum=3)
    # calc = GPAW(mode=PW(700), xc='HSE06', charge=4)
    # atoms.calc = calc
    # atoms.get_potential_energy()

    calc = GPAW(h=0.12, mode='lcao', basis='qzdp',
                xc='HSE06WIP:backend=ri', charge=4)
    atoms.calc = calc
    atoms.get_potential_energy()

    # TURBOMOLE C^4+ calculation aug-cc-pV5Z
    tm_eigs = [-68.5345, -61.6901]
    tm_e = -32.35980873243 * Hartree

    assert np.allclose(tm_eigs, calc.get_eigenvalues()[:2], atol=0.1)
    assert np.allclose(tm_e, calc.hamiltonian.setups.Eref * Hartree +
                       atoms.get_potential_energy(), atol=0.1)

    atoms = bulk('C', 'diamond')
    atoms.calc = GPAW(kpts={'size': (2, 2, 2), 'gamma': True},
                      mode='lcao', basis='dzp',
                      xc='HSE06WIP:backend=ri')

    # NOTE: HSE06 does not yet work. This is just a placeholder for
    # integration test.
    assert np.allclose(atoms.get_potential_energy(), 3.291175, atol=4e-5)

    calc = atoms.calc

    setup = calc.density.setups[0]
    # With w=0 the values should agree exactly. For small systems
    # with small paw-cutoffs, the values are very close nevertheless
    # Thus assert with a moderate tolerance.
    assert np.allclose(setup.M_pp,
                       setup.M_wpp[0.11],
                       rtol=1e-2, atol=1e-4)

    assert np.allclose(setup.X_p, setup.X_wp[0.11], atol=1e-5)
