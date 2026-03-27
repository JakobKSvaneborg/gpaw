from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.move_wfs import move_wave_functions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


class PWFDIBZWaveFunctions(IBZWaveFunctions[PWFDWaveFunctions]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_wave_functions = move_wave_functions

        # Allow sharing of PAW-projectors between spin up and down:
        if self.nspins == 2:
            for wfs1, wfs2 in zip(self._wfs_u[:-1], self._wfs_u[1:]):
                if wfs1.q == wfs2.q:
                    wfs1.other_spin = wfs2

    def has_wave_functions(self):
        return self._wfs_u[0].psit_nX.data is not None
