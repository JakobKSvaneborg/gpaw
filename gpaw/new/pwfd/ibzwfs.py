from gpaw.new.ibzwfs import IBZWaveFunctions


class PWFDIBZWaveFunction(IBZWaveFunctions):
    def move(self, fracpos_ac, atomdist):
        super().move(fracpos_ac, atomdist)
        for wfs in self:
            wfs.move(fracpos_ac, atomdist)

    @property
    def desc(self):
        return self.wfs_qs[0][0].psit_nX.desc
