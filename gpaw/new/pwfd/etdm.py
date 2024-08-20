from gpaw.new.backwards_compatibility import (FakeDensity, FakeHamiltonian,
                                              FakeWFS)
from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian


class ETDMPWFD(Eigensolver):
    def __init__(self, atoms, params):
        from gpaw.directmin.etdm_fdpw import FDPWETDM
        self.eigensolver = FDPWETDM(**params)
        self.atoms = atoms

    def iterate(self, state: DFTState, hamiltonian: Hamiltonian) -> float:
        wfs = FakeWFS(state, self.atoms)
        ham = FakeHamiltonian(state)
        dens = FakeDensity(state)
        if not self.eistgensolver.initialized:
            self.eigensolver.initialize_dm_helper(wfs, ham, dens, print)
        wfs.eigensolver.iterate(ham, wfs, dens, print)
        assert not wfs.eigensolver.check_restart(wfs)
        e_entropy = 0.0
        kin_en_using_band = False

        # ham.get_energy(
        #     e_entropy, wfs, kin_en_using_band=kin_en_using_band, e_sic=e_sic)
