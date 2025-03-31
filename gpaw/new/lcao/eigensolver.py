import numpy as np

from gpaw.new.eigensolver import Eigensolver, calculate_weights
from gpaw.new.lcao.hamiltonian import HamiltonianMatrixCalculator
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.energies import DFTEnergies


class LCAOEigensolver(Eigensolver):
    def __init__(self,
                 basis,
                 converge_bands='occupied'):
        self.basis = basis
        self.converge_bands = converge_bands

    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian,
                pot_calc=None,
                energies=None) -> tuple[float, DFTEnergies]:
        matrix_calculator = hamiltonian.create_hamiltonian_matrix_calculator(
            potential)

        weight_un = calculate_weights(self.converge_bands, ibzwfs)
        eig_error = 0.0
        for wfs, weight_n in zip(ibzwfs, weight_un):
            has_eigs = True
            try:
                eig_old = wfs.myeig_n
            except ValueError:  # no eigenvalues yet
                eig_old = np.inf
                has_eigs = False
            self.iterate1(wfs, matrix_calculator)
            if has_eigs:
                eig_error += weight_n @ np.abs(eig_old - wfs.myeig_n)**2
            else:  # no eigenvalues yet
                eig_error = np.inf
        
        eig_error = (ibzwfs.kpt_band_comm.sum_scalar(
                     float(eig_error)) * ibzwfs.spin_degeneracy)**0.5
        return eig_error, 0.0, energies

    def iterate1(self,
                 wfs: LCAOWaveFunctions,
                 matrix_calculator: HamiltonianMatrixCalculator):
        H_MM = matrix_calculator.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        C_Mn = H_MM  # rename (H_MM now contains the eigenvectors)
        assert len(eig_M) >= wfs.nbands
        N = wfs.nbands
        wfs._eig_n = np.empty(wfs.nbands)
        wfs._eig_n[:] = eig_M[:N]
        comm = C_Mn.dist.comm
        if comm.size == 1:
            wfs.C_nM.data[:] = C_Mn.data.T[:N]
        else:
            C_Mn = C_Mn.gather(broadcast=True)
            n1, n2 = wfs.C_nM.dist.my_row_range()
            wfs.C_nM.data[:] = C_Mn.data.T[n1:n2]

        # Make sure wfs.C_nM and (lazy) wfs.P_ani are in sync:
        wfs._P_ani = None
