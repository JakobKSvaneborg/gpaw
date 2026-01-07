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

    def pwify(self, relpos_ac, setups, basis_set):
        from gpaw.core.plane_waves import PWDesc
        from gpaw.core.pwacf import PWAtomCenteredFunctions

        pw0 = PWDesc(ecut=self.grid.ekin_max(),
                     cell=self.grid.cell,
                     comm=self.grid.comm,
                     dtype=self.dtype)
        for wfs in self:
            pw = pw0.new(kpt=wfs.kpt_c)
            if 1:
                grid = self.grid.new(kpt=wfs.kpt_c, dtype=self.dtype)
                emikr_R = grid.eikr(-wfs.kpt_c)
                psit_nG = pw.empty(self.nbands, self.band_comm)
                psit_nR = grid.zeros(self.nbands)
                basis_set.lcao_to_grid(wfs.C_nM.data, psit_nR.data, wfs.q)

                for psit_R, psit_G in zip(psit_nR, psit_nG):
                    psit_R.data *= emikr_R
                    psit_R.fft(out=psit_G)
            else:
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
            print(wfs.kpt_c)
            wfs.psit_nX = psit_nG
            pt_aiG = psit_nG.desc.atom_centered_functions(
                [setup.pt_j for setup in setups],
                relpos_ac)
            P_ani = pt_aiG.integrate(psit_nG)
            print('P', P_ani[0])
            print('P', wfs.P_ani[0])

