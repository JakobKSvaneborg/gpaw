import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.cdft.cdft import CDFT


@pytest.mark.old_gpaw_only
def test_cdft_forces_consistency(in_tmp_dir):
    delta = 0.01
    e, f = cdft(0.0)
    if 0:
        eplus, _ = cdft(delta)
        eminus, _ = cdft(-delta)
        f_finite_difference = (eminus - eplus) / (2 * delta)
    else:
        f_finite_difference = -17.98
    assert f == pytest.approx(f_finite_difference, abs=0.01)


def cdft(dz):
    charge_regions = [[0], [1]]
    charge = [0.4, -0.2]
    coefs = [0, 0]
    atoms = Atoms('COO',
                  [[3.1, 2.98, 3.12 + dz],
                   [2.92, 3.00, 4.25],
                   [2.95, 2.97, 1.83]],
                  cell=[6, 6, 6])
    calc_ground = GPAW(legacy_gpaw=True,
                       mode='fd',
                       spinpol=True,
                       txt='ground_state_output_A.txt')
    calc_cdft = CDFT(calc=calc_ground,
                     atoms=atoms,
                     charge_regions=charge_regions,
                     charges=charge,
                     charge_coefs=coefs,
                     method='L-BFGS-B',
                     txt='cdftA_output.txt',
                     minimizer_options={'gtol': 0.001})
    atoms.calc = calc_cdft
    e_cdft = atoms.get_potential_energy()
    f_cdft = atoms.get_forces()[0, 2]
    return e_cdft, f_cdft
