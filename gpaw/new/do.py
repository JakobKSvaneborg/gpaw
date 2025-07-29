from gpaw.dft import Eigensolver
from gpaw.new.pwfd.etdm import ETDM


class DirectOptimization(Eigensolver):
    def __init__(self,
                 converge_unocc: bool = False):
        self.converge_unocc = converge_unocc

    def todict(self):
        return {'converge_unocc': self.converge_unocc}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              hamiltonian,
              converge_bands,
              setups,
              atoms):
        return ETDM(converge_unocc=self.converge_unocc)
