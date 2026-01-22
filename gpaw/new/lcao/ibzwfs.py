from math import pi

import numpy as np
from gpaw.new.density import Density
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


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

    def convert_to(self,
                   mode: str,
                   grid=None,
                   pw=None,
                   qspiral_v=None,
                   nbands: int | None = None) -> PWFDIBZWaveFunctions:
        nbands = nbands or self.nbands

        def create_wfs(spin, q, k, kpt_c, weight):
            lcaowfs = self._get_wfs(k, spin)
            assert lcaowfs.spin == spin

            if mode == 'pw':
                psit_nX = lcaowfs.to_pw_expansion(nbands, pw)
            elif mode == 'fd':
                psit_nX = lcaowfs.to_uniform_grid(nbands, grid)
            else:
                raise ValueError(f'Illegal mode: {mode}')

            mylcaonbands, nao = lcaowfs.C_nM.dist.shape
            mynbands = len(psit_nX.data)
            eig_n = np.empty(nbands)
            eig_n[:self.nbands] = lcaowfs.eig_n
            eig_n[self.nbands:] = 100.0  # set high value for random wfs.
            if mylcaonbands < mynbands:
                psit_nX[mylcaonbands:].randomize(
                    seed=self.comm.rank)
            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=lcaowfs.setups,
                relpos_ac=lcaowfs.relpos_ac,
                atomdist=lcaowfs.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=qspiral_v)
            wfs.eig_n = eig_n
            if lcaowfs.has_occs:
                wfs._occ_n = lcaowfs._occ_n.copy()
            return wfs

        return PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.kpt_comm,
            kpt_band_comm=self.kpt_band_comm,
            comm=self.comm)
