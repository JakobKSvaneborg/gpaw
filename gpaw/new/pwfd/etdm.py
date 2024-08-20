from gpaw.new.eigensolver import Eigensolver
from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.ibzwfs import IBZWaveFunctions


class ETDMPWFD:
    def __init__(self, eigensolver):
        self.eigensolver = eigensolver

    def iterate(self, state: DFTState, hamiltonian: Hamiltonian) -> float:
        if not self.eigensolver.initialized:
            self.eigensolver.initialize_dm_helper(wfs, ham, dens, log)
            wfs.eigensolver.iterate(ham, wfs, dens, log)
            restart = wfs.eigensolver.check_restart(wfs)
            e_entropy = 0.0
            kin_en_using_band = False
        else:
            wfs.eigensolver.iterate(ham, wfs)
            e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            kin_en_using_band = True

        if hasattr(wfs.eigensolver, 'e_sic'):
            e_sic = wfs.eigensolver.e_sic
        else:
            e_sic = 0.0

        ham.get_energy(
            e_entropy, wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic)
