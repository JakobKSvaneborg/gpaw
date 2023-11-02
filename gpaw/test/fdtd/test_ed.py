import pytest
from gpaw import GPAW
from gpaw.tddft import TDDFT, DipoleMomentWriter
from gpaw.test import equal


@pytest.mark.later
def test_fdtd_ed(in_tmp_dir, gpw_files):
    # Accuracy
    energy_eps = 0.0005

    # load gpw file
    gs_calc = GPAW(gpw_files['na2_isolated'])
    energy = gs_calc.get_potential_energy()

    # Test ground state
    equal(energy, -0.631881, energy_eps * gs_calc.get_number_of_electrons())

    # Test floating point arithmetic errors
    equal(gs_calc.hamiltonian.poisson.shift_indices_1, [4, 4, 10], 0)
    equal(gs_calc.hamiltonian.poisson.shift_indices_2, [8, 8, 16], 0)

    # Initialize TDDFT and FDTD
    kick = [0.0, 0.0, 1.0e-3]
    time_step = 10.0
    max_time = 100  # 0.1 fs

    td_calc = TDDFT(gpw_files['na2_isolated'])
    DipoleMomentWriter(td_calc, 'dm.dat')
    td_calc.absorption_kick(kick_strength=kick)

    # Propagate TDDFT and FDTD
    td_calc.propagate(time_step, max_time / time_step / 2)
    td_calc.write('td.gpw', 'all')

    td_calc2 = TDDFT('td.gpw')
    DipoleMomentWriter(td_calc2, 'dm.dat')
    td_calc2.propagate(time_step, max_time / time_step / 2)

    # Test
    ref_cl_dipole_moment = [5.25374117e-14, 5.75811267e-14, 3.08349334e-02]
    ref_qm_dipole_moment = [1.78620337e-11, -1.57782578e-11, 5.21368300e-01]

    tol = 1e-4
    equal(td_calc2.hamiltonian.poisson.get_classical_dipole_moment(),
          ref_cl_dipole_moment, tol)
    equal(td_calc2.hamiltonian.poisson.get_quantum_dipole_moment(),
          ref_qm_dipole_moment, tol)
