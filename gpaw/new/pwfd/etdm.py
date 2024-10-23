import numpy as np

from gpaw.new.backwards_compatibility import FakeHamiltonian, FakeWFS
from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.directmin.scf_helper import check_eigensolver_state, do_if_converged


class ETDMPWFD(Eigensolver):
    direct = True

    def __init__(self, setups, comm, atoms, eigensolver):
        self.eigensolver = eigensolver
        self.eigensolver.gpaw_new = True
        self.setups = setups
        self.comm = comm
        self.atoms = atoms
        self.pot_calc = None
        self.occ_calc = None
        self.log = None

    def whd(self, state, hamiltonian):
        wfs = FakeWFS(state, self.setups, self.comm, self.occ_calc,
                      hamiltonian,
                      self.atoms)
        wfs.eigensolver = self.eigensolver
        ham = FakeHamiltonian(state, self.pot_calc)
        dens = Density(state)
        return wfs, ham, dens

    def initialize_etdm(self, state, pot_calc, occ_calc, hamiltonian, mixer,
                        log):
        self.pot_calc = pot_calc
        self.occ_calc = occ_calc
        self.log = log
        oldwfs, ham, dens = self.whd(state, hamiltonian)
        dens.mixer = mixer
        for wfs in state.ibzwfs:
            if wfs._eig_n is None:
                wfs._eig_n = np.empty(wfs.nbands)
        check_eigensolver_state('etdm-fdpw', oldwfs, ham, dens, log=log)

    def iterate(self, state: DFTState,
                hamiltonian: Hamiltonian) -> float:
        wfs, ham, dens = self.whd(state, hamiltonian)
        if not self.eigensolver.initialized:
            self.eigensolver.initialize_dm_helper(wfs, ham, dens, self.log)
        self.eigensolver.iterate(ham, wfs, dens, self.log)
        assert not self.eigensolver.check_restart(wfs)
        e_entropy = 0.0
        kin_en_using_band = False
        e_sic = 0.0
        ham.get_energy(
            e_entropy, wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic)
        return self.eigensolver.error

    def postprocess(self, state, hamiltonian):
        wfs, ham, dens = self.whd(state, hamiltonian)
        do_if_converged(
            'etdm-fdpw', wfs, ham, dens, self.log)


class Density:
    def __init__(self, state):
        self.fixed = False
        self.state = state

    def update(self, wfs):
        self.state.density.update(self.state.ibzwfs)
