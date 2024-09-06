from gpaw.new.ibzwfs import IBZWaveFunctions


class PWFDIBZWaveFunction(IBZWaveFunctions):
    def move(self, fracpos_ac, atomdist):
        super().move(fracpos_ac, atomdist)
        for wfs in self:
            wfs.move(fracpos_ac, atomdist)

    def has_wave_functions(self):
        return self.wfs_qs[0][0].psit_nX.data is not None
