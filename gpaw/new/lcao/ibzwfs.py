from math import pi

from gpaw.new.density import Density
from gpaw.new.ibzwfs import IBZWaveFunctions


class LCAOIBZWaveFunctions(IBZWaveFunctions):
    def has_wave_functions(self):
        return True

    def move(self, relpos_ac, atomdist):
        from gpaw.new.lcao.builder import tci_helper

        super().move(relpos_ac, atomdist)

        for wfs in self:
            basis = wfs.basis
            setups = wfs.setups
            break
        basis.set_positions(relpos_ac)
        myM = (basis.Mmax + self.band_comm.size - 1) // self.band_comm.size
        basis.set_matrix_distribution(
            min(self.band_comm.rank * myM, basis.Mmax),
            min((self.band_comm.rank + 1) * myM, basis.Mmax))

        S_qMM, T_qMM, P_qaMi, tciexpansions, tci_derivatives = tci_helper(
            basis, self.ibz, self.domain_comm, self.band_comm, self.kpt_comm,
            relpos_ac, atomdist,
            self.grid, self.dtype, setups, self.xp,
            nspins=self.nspins)

        for wfs in self:
            wfs.tci_derivatives = tci_derivatives
            wfs.S_MM = S_qMM[wfs.q]
            wfs.T_MM = T_qMM[wfs.q]
            wfs.P_aMi = P_qaMi[wfs.q]

    def normalize_density(self, density: Density) -> None:
        """Normalize density.

        Basis functions may extend outside box!
        """
        pseudo_charge = density.nt_sR.integrate().sum()
        ccc_aL = density.calculate_compensation_charge_coefficients()
        comp_charge = (4 * pi)**0.5 * sum(float(ccc_L[0])
                                          for ccc_L in ccc_aL.values())
        comp_charge = ccc_aL.layout.atomdist.comm.sum_scalar(comp_charge)
        density.nt_sR.data *= -comp_charge / pseudo_charge

    def pwify(self, relpos_ac, setups):
        from gpaw.core.plane_waves import PWDesc
        from gpaw.core.pwacf import PWAtomCenteredFunctions

        pw0 = PWDesc(ecut=self.grid.ekin_max(),
                     cell=self.grid.cell,
                     comm=self.grid.comm,
                     dtype=self.dtype)
        for wfs in self:
            pw = pw0.new(kpt=wfs.kpt_c)
            phit_aJG = PWAtomCenteredFunctions(
                [setup.basis_functions_J for setup in setups],
                relpos_ac,
                pw,
                atomdist=wfs.atomdist,
                xp=self.xp)
            psit_nG = pw.empty(self.nbands,
                               comm=self.band_comm,
                               xp=self.xp)
            mynbands, M = wfs.C_nM.dist.shape
            phit_aJG.multiply(wfs.C_nM.to_dtype(pw.dtype),
                              out_nG=psit_nG[:mynbands])
            wfs.psit_nX = psit_nG
            print(psit_nG)
