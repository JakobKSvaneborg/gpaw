from gpaw.dft import Eigensolver, LegacyGPAWError
from gpaw.new.pwfd.diropt import DirOptPWFD


class DirectOptimization(Eigensolver):
    name = 'etdm-fdpw'

    def __init__(self,
                 converge_unocc: bool = False,
                 **kwargs):
        if kwargs:
            raise LegacyGPAWError
        self.converge_unocc = converge_unocc

    def todict(self):
        return {'converge_unocc': self.converge_unocc}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              hamiltonian,
              convergence,
              setups,
              atoms):
        return DirOptPWFD(
            converge_unocc=self.converge_unocc,
            converge_bands=convergence,
            hamiltonian=hamiltonian)
