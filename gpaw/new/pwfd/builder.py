from types import SimpleNamespace
from math import pi

import numpy as np

from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.pwfd.eigensolver import create_eigensolver as make_eigensolver
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


class PWFDDFTComponentsBuilder(DFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 *,
                 comm,
                 dtype=None,
                 qspiral=None):
        super().__init__(atoms, params, dtype=dtype, comm=comm)
        self.qspiral_v = (None if qspiral is None else
                          qspiral @ self.grid.icell * (2 * pi))

    def create_eigensolver(self, hamiltonian):
        return make_eigensolver(
            self.nbands,
            self.wf_desc,
            self.communicators['b'],
            self.communicators['w'],
            hamiltonian.create_preconditioner,
            self.params.convergence.get('bands', 'occupied'),
            self.setups,
            self.atoms,
            **self.params.eigensolver)

    def read_ibz_wave_functions(self, reader, log):
        kpt_comm, band_comm, domain_comm = (self.communicators[x]
                                            for x in 'kbd')

        def create_wfs(spin: int, q: int, k: int, kpt_c, weight: float):
            psit_nG = SimpleNamespace(
                comm=band_comm,
                dims=(self.nbands,),
                desc=self.wf_desc.new(kpt=kpt_c),
                data=None,
                xp=np)
            wfs = PWFDWaveFunctions(
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                psit_nX=psit_nG,  # type: ignore
                setups=self.setups,
                relpos_ac=self.relpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)

            return wfs

        ibzwfs = PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])

        # Set eigenvalues, occupations, etc..
        self.read_wavefunction_values(reader, ibzwfs)

        return ibzwfs

    def create_ibz_wave_functions(self, basis, potential, *, log):
        from gpaw.new.lcao.builder import create_lcao_ibzwfs

        if self.params.random:
            return self.create_random_ibz_wave_functions(log)

        # sl_default = self.params.parallel['sl_default']
        # sl_lcao = self.params.parallel['sl_lcao'] or sl_default

        lcaonbands = min(self.nbands,
                         basis.Mmax * (2 if self.ncomponents == 4 else 1))
        lcao_ibzwfs, _ = create_lcao_ibzwfs(
            basis,
            self.ibz, self.communicators, self.setups,
            self.relpos_ac, self.grid, self.dtype,
            lcaonbands, self.ncomponents, self.atomdist, self.nelectrons)

        hamiltonian = LCAOHamiltonian(basis)
        LCAOEigensolver(basis).iterate(
            lcao_ibzwfs, None, potential, hamiltonian)

        def create_wfs(spin, q, k, kpt_c, weight):
            lcaowfs = lcao_ibzwfs.wfs_qs[q][spin]
            assert lcaowfs.spin == spin

            # Convert to PW-coefs in PW-mode:
            psit_nX = self.convert_wave_functions_from_uniform_grid(
                lcaowfs.C_nM, basis, kpt_c, q)

            mylcaonbands, nao = lcaowfs.C_nM.dist.shape
            mynbands = len(psit_nX.data)
            eig_n = np.empty(self.nbands)
            eig_n[:lcaonbands] = lcaowfs._eig_n
            eig_n[lcaonbands:] = 100.0  # set high value for random wfs.
            if mylcaonbands < mynbands:
                psit_nX[mylcaonbands:].randomize(
                    seed=self.communicators['w'].rank)

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                relpos_ac=self.relpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)
            wfs._eig_n = eig_n
            return wfs

        return PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])

    def create_random_ibz_wave_functions(self, log):
        log('Initializing wave functions with random numbers')

        def create_wfs(spin, q, k, kpt_c, weight):
            desc = self.wf_desc.new(kpt=kpt_c)
            psit_nX = desc.empty(
                dims=(self.nbands,),
                comm=self.communicators['b'],
                xp=self.xp)
            psit_nX.randomize()

            wfs = PWFDWaveFunctions(
                psit_nX=psit_nX,
                spin=spin,
                q=q,
                k=k,
                weight=weight,
                setups=self.setups,
                relpos_ac=self.relpos_ac,
                atomdist=self.atomdist,
                ncomponents=self.ncomponents,
                qspiral_v=self.qspiral_v)

            return wfs

        return PWFDIBZWaveFunctions.create(
            ibz=self.ibz,
            nelectrons=self.nelectrons,
            ncomponents=self.ncomponents,
            create_wfs_func=create_wfs,
            kpt_comm=self.communicators['k'],
            kpt_band_comm=self.communicators['D'],
            comm=self.communicators['w'])
