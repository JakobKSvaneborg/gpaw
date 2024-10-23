from gpaw.new.ibzwfs import IBZWaveFunctions


class PWFDIBZWaveFunction(IBZWaveFunctions):
    def has_wave_functions(self):
        return self.wfs_qs[0][0].psit_nX.data is not None
