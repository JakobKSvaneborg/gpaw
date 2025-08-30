# MQ: cores=4
"""Check non self-consistent SOC calculation with and without symmetry."""
from typing import Dict
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.spinorbit import soc_eigenstates

atoms = Atoms('Fe',
              magmoms=[2.5],
              cell=[2.3, 5, 5],
              pbc=True)


def check(array1_mx, array2_mx, k, tol=1e-6):
    m = 0
    for a1_x, a2_x in zip(array1_mx, array2_mx):
        b1_y = a1_x.ravel()
        b2_y = a2_x.ravel()
        y = abs(b1_y).argmax()
        assert abs(b1_y - b2_y * b1_y[y] / b2_y[y]).max() < tol
        m += 1


def soc(params: Dict) -> list:
    """Do DFT + SOC calculations in memory and from gpw-file."""
    name = 'Fe-sym-' + params.get('symmetry', 'on')
    atoms.calc = GPAW(txt=name + '.txt', **params)
    atoms.get_potential_energy()
    s1 = soc_eigenstates(atoms.calc)
    atoms.calc.write(name + '.gpw')
    s2 = soc_eigenstates(name + '.gpw')
    assert abs(s1.fermi_level - s2.fermi_level) < 1e-8
    assert abs(s1.eigenvalues() - s2.eigenvalues()).max() < 1e-6
    p1 = s1.spin_projections()
    p2 = s2.spin_projections()
    p12 = p1 - p2
    p12[3] = 0.0
    assert abs(p12).max() < 1e-7
    for wf1, wf2 in zip(s1, s2):
        assert wf1.bz_index == wf2.bz_index
        P1_msI = wf1.projections.array
        P2_msI = wf2.projections.array
        if wf1.bz_index != 3:
            check(P1_msI, P2_msI, wf1.bz_index)
    return s1, atoms.calc


def go() -> None:
    """Compare with and without symmetry."""
    params = dict(mode=PW(500),
                  xc='PBE',
                  kpts=[7, 1, 1],
                  convergence={'eigenstates': 1e-10,
                               'bands': 9})
    A, _ = soc(params)
    params['symmetry'] = 'off'
    B, calc = soc(params)
    assert abs(A.fermi_level - B.fermi_level) < 0.002
    assert abs(A.eigenvalues() - B.eigenvalues()).max() < 0.003
    p1 = A.spin_projections()
    p2 = B.spin_projections()
    assert abs(p1 - p2).max() < 0.15


def workflow():
    from myqueue.workflow import run
    run(function=go, cores=4)


if __name__ == '__main__':
    go()
