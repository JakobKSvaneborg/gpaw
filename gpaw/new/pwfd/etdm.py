from gpaw.new.backwards_compatibility import FakeHamiltonian, FakeWFS
from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian


class ETDMPWFD(Eigensolver):
    def __init__(self, setups, comm, atoms, params):
        from gpaw.directmin.etdm_fdpw import FDPWETDM
        self.eigensolver = FDPWETDM(**params)
        self.eigensolver.gpaw_new = True
        self.setups = setups
        self.comm = comm
        self.atoms = atoms
        self.pot_calc = None
        self.occ_calc = None

    def iterate(self, state: DFTState,
                hamiltonian: Hamiltonian) -> float:
        wfs = FakeWFS(state, self.setups, self.comm, self.occ_calc,
                      hamiltonian,
                      self.atoms)
        ham = FakeHamiltonian(state, self.pot_calc)
        dens = Density(state)
        if not self.eigensolver.initialized:
            self.eigensolver.initialize_dm_helper(wfs, ham, dens, print)
        self.eigensolver.iterate(ham, wfs, dens, print)
        assert not self.eigensolver.check_restart(wfs)
        e_entropy = 0.0
        kin_en_using_band = False
        e_sic = 0.0
        ham.get_energy(
            e_entropy, wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic)
        return self.eigensolver.error


class Density:
    def __init__(self, state):
        self.fixed = False
        self.state = state

    def update(self, wfs):
        self.state.density.update(self.state.ibzwfs)
