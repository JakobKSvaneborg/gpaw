
import pytest
from ase import Atoms
from gpaw.new.symmetry import (Symmetries, SymmetryAnalysisBug,
                               create_symmetries_object)


def test_incomplete_point_group1():

    cell_cv = [[3.0787358319029257, 0.0, 0.0],
               [-1.4504494987687535, 2.7156602464528716, 0.0],
               [-1.4504494987687535, -0.976306364588358, 2.534094800244998]]

    kwargs = {'cell': cell_cv,
              'pbc': (1, 1, 1),
              'tolerance': 1e-1,
              'relative_positions': [[0.5, 0.75, 0.25],
                                     [0.5, 0.25, 0.75],
                                     [0., 0., 0.]],
              'ids': [0, 0, 1],
              'symmorphic': True,
              '_backwards_compatible': False}

    with pytest.raises(SymmetryAnalysisBug):
        sym = Symmetries.from_cell_and_atoms(**kwargs, guarantee_group=False)
    sym = Symmetries.from_cell_and_atoms(**kwargs, guarantee_group=True)
    assert len(sym) == 16


def test_incomplete_point_group2():

    cell_cv = [[6.484107793708888, 0.021988810132164173, 0.033790665556415],
               [-2.725837683619327, 5.883358353028015, -0.03379073060296549],
               [-2.7258376201572068, -2.4138174313122733, 5.365494543828883]]
    position_av = [[-1.36291905, 0.86738539, 1.33292615],
                   [-4.08875625, 2.60215553, 3.99877766],
                   [-1.90903349, 0.04228007, 4.60240183],
                   [1.87913321, -0.023255, 1.93654713],
                   [-0.84670072, 3.51478474, 3.42894735],
                   [2.94146598, 3.44924966, 0.76309265]]

    atoms = Atoms('Sr2Te4', cell=cell_cv, positions=position_av, pbc=True)

    kwargs = {'tolerance': 10**(-5.45), 'symmorphic': True,
              '_backwards_compatible': False}

    with pytest.raises(SymmetryAnalysisBug):
        sym = create_symmetries_object(atoms, **kwargs, guarantee_group=False)
    sym = create_symmetries_object(atoms, **kwargs, guarantee_group=True)
    assert len(sym) == 2
