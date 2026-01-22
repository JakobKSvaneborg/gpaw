from gpaw.dft import Eigensolver
from gpaw.new.pwfd.diropt import DirOptPWFD


class DirectOptimization(Eigensolver):
    name = 'etdm-fdpw'

    def __init__(self,
                 converge_unocc: bool = False,
                 **kwargs):
        if kwargs:
            raise NotImplementedError
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
        return DirOptPWFD(
            converge_unocc=self.converge_unocc,
            hamiltonian=hamiltonian)
