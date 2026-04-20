import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress

from gpaw import GPAW, PW, Mixer
from gpaw.mpi import world
from gpaw.test import gen
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.xc.libvdwxc import libvdwxc_has_spin, libvdwxc_has_stress, vdw_df

skip_reason = "libvdwxc version does not implement stress"
if compiled_with_libvdwxc():
    try:
        skip_cond = not libvdwxc_has_stress()
    except SystemError:
        skip_cond = True
else:
    skip_cond = True


def _check_libvdwxc_stress(in_tmp_dir, xc, setups, s_ref):
    assert libvdwxc_has_spin() and libvdwxc_has_stress()
    si = bulk('Si')
    si.calc = GPAW(mode=PW(200),
                   mixer=Mixer(0.7, 5, 50.0),
                   xc=xc,
                   kpts=(1, 1, 2),  # Run (1, 1, 2) to avoid gamma pt code
                   convergence={'energy': 1e-8},
                   parallel={'domain': min(2, world.size)},
                   setups=setups,
                   txt='si_stress.txt')

    si.set_cell(np.dot(si.cell,
                       [[1.02, 0, 0.03],
                        [0, 0.99, -0.02],
                        [0.2, -0.01, 1.03]]),
                scale_atoms=True)

    si.get_potential_energy()

    s_analytical = si.get_stress()
    if s_ref is None:
        s_ref = calculate_numerical_stress(si, 1e-5)
        print("REF", s_ref)
    print(s_analytical)
    s_err = s_analytical - s_ref
    assert np.all(abs(s_err) < 1e-4)


@pytest.mark.stress
@pytest.mark.skipif(skip_cond, reason=skip_reason)
def test_pw_si_stress_libvdwxc_gga(in_tmp_dir):
    s_ref = [-0.17538969, -0.08992427, -0.14401403,
             -0.05984851, -0.00058295, 0.04351124]
    xc = vdw_df()
    _check_libvdwxc_stress(in_tmp_dir, xc, "paw", s_ref)


@pytest.mark.stress
@pytest.mark.skipif(skip_cond, reason=skip_reason)
def test_pw_si_stress_libvdwxc_mgga(in_tmp_dir):
    setups = {'Si': gen('Si', xcname='PBEsol')}
    s_ref = [-0.13724147, -0.05995537, -0.11039066,
             -0.05893786, 0.01830188, 0.04031258]
    xc = {'name': 'mBEEF-vdW', 'backend': 'libvdwxc'}
    _check_libvdwxc_stress(in_tmp_dir, xc, setups, s_ref)
