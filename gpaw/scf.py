import time
import warnings

import numpy as np
from ase.units import Ha

from gpaw import KohnShamConvergenceError
from gpaw.convergence_criteria import check_convergence
from gpaw.forces import calculate_forces


class SCFLoop:
    """Self-consistent field loop."""
    def __init__(self, criteria, maxiter=100, niter_fixdensity=None):
        self.criteria = criteria
        self.maxiter = maxiter
        self.niter_fixdensity = niter_fixdensity
        self.niter = None
        self.reset()
        self.converged = False
        self.eigensolver_used = None

    def __str__(self):
        s = 'Convergence criteria:\n'
        for criterion in self.criteria.values():
            if criterion.description is not None:
                s += ' ' + criterion.description + '\n'
        s += f' Maximum number of scf [iter]ations: {self.maxiter}'
        s += ("\n (Square brackets indicate name in SCF output, whereas a 'c'"
              " in\n the SCF output indicates the quantity has converged.)\n")
        return s

    def write(self, writer):
        writer.write(converged=self.converged)

    def read(self, reader):
        if reader.version < 4:
            self.converged = reader.scf.converged
        else:
            self.converged = True

    def reset(self):
        for criterion in self.criteria.values():
            criterion.reset()
        self.converged = False
        self.eigensolver_used = None

    def irun(self, wfs, ham, dens, log, callback):

        self.eigensolver_used = getattr(wfs.eigensolver, "name", None)
        self.check_eigensolver_state(wfs, ham, dens)
        self.niter = 1
        converged = False

        while self.niter <= self.maxiter:
            self.iterate_eigensolver(wfs, ham, dens)

            ctx = self.check_convergence(
                dens, ham, wfs, log, callback)
            yield ctx

            converged = (self.converged and
                         self.niter >= self.niter_fixdensity)
            if converged:
                self.do_if_converged(wfs, ham, dens, log)
                break

            self.update_ham_and_dens(wfs, ham, dens)
            self.niter += 1

        # Don't fix the density in the next step.
        self.niter_fixdensity = 0

        if not converged:
            self.not_converged(dens, ham, wfs, log)

    def log(self, log, converged_items, entries, context):
        """Output from each iteration."""
        write_iteration(self.criteria, converged_items, entries, context, log)

    def check_convergence(self, dens, ham, wfs, log, callback):

        context = SCFEvent(dens=dens, ham=ham, wfs=wfs, niter=self.niter,
                           log=log)

        # Converged?
        # entries: for log file, per criteria
        # converged_items: True/False, per criteria
        self.converged, converged_items, entries = check_convergence(
            self.criteria, context)

        callback(self.niter)
        self.log(log, converged_items, entries, context)
        return context

    def not_converged(self, dens, ham, wfs, log):

        context = SCFEvent(dens=dens, ham=ham, wfs=wfs, niter=self.niter,
                           log=log)
        eigerr = self.criteria['eigenstates'].get_error(context)
        if not np.isfinite(eigerr):
            msg = 'Not enough bands for ' + wfs.eigensolver.nbands_converge
            log(msg)
            log.fd.flush()
            raise KohnShamConvergenceError(msg)
        log(oops, flush=True)
        raise KohnShamConvergenceError(
            'Did not converge!  See text output for help.')

    def check_eigensolver_state(self, wfs, ham, dens):

        if self.eigensolver_used == 'etdm':
            wfs.eigensolver.eg_count = 0
            wfs.eigensolver.globaliters = 0
            wfs.eigensolver.check_assertions(wfs, dens)
            if wfs.eigensolver.dm_helper is None:
                wfs.eigensolver.initialize_dm_helper(wfs, ham, dens)

    def iterate_eigensolver(self, wfs, ham, dens):

        if self.eigensolver_used == 'etdm':
            wfs.eigensolver.iterate(ham, wfs, dens)
            wfs.eigensolver.check_mom(wfs, dens)
            e_entropy = 0.0
            kin_en_using_band = False
        else:
            wfs.eigensolver.iterate(ham, wfs)
            e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            kin_en_using_band = True

        ham.get_energy(e_entropy, wfs, kin_en_using_band=kin_en_using_band)

    def do_if_converged(self, wfs, ham, dens, log):

        if self.eigensolver_used == 'etdm':
            energy = ham.get_energy(0.0, wfs, kin_en_using_band=False)
            wfs.calculate_occupation_numbers(dens.fixed)
            wfs.eigensolver.get_canonical_representation(
                ham, wfs, dens, sort_eigenvalues=True)
            energy_converged = wfs.eigensolver.update_ks_energy(ham, wfs, dens)
            energy_diff_after_scf = abs(energy - energy_converged) * Ha
            if energy_diff_after_scf > 1.0e-6:
                warnings.warn('Jump in energy of %f eV detected at the end of '
                              'SCF after getting canonical orbitals, SCF '
                              'might have converged to the wrong solution '
                              'or achieved energy convergence to the correct '
                              'solution above 1.0e-6 eV'
                              % (energy_diff_after_scf))

            log('\nOccupied states converged after'
                ' {:d} e/g evaluations'.format(wfs.eigensolver.eg_count))

    def update_ham_and_dens(self, wfs, ham, dens):

        to_update = self.niter > self.niter_fixdensity and not dens.fixed
        if self.eigensolver_used == 'etdm' or not to_update:
            ham.npoisson = 0
        else:
            dens.update(wfs)
            ham.update(dens)


def write_iteration(criteria, converged_items, entries, ctx, log):
    custom = (set(criteria) -
              {'energy', 'eigenstates', 'density'})

    if ctx.niter == 1:
        header1 = ('     {:<4s} {:>8s} {:>12s}  '
                   .format('iter', 'time', 'total'))
        header2 = ('     {:>4s} {:>8s} {:>12s}  '
                   .format('', '', 'energy'))
        header1 += 'log10-change:'
        header2 += ' eigst   dens  '
        for name in custom:
            criterion = criteria[name]
            header1 += ' ' * 7
            header2 += f'{criterion.tablename:>5s}  '
        if ctx.wfs.nspins == 2:
            header1 += '{:>8s} '.format('magmom')
            header2 += '{:>8s} '.format('')
        log(header1.rstrip())
        log(header2.rstrip())

    c = {k: 'c' if v else ' ' for k, v in converged_items.items()}

    # Iterations and time.
    now = time.localtime()
    line = ('iter:{:4d} {:02d}:{:02d}:{:02d} '
            .format(ctx.niter, *now[3:6]))

    # Energy.
    line += '{:>12s}{:1s} '.format(entries['energy'], c['energy'])

    # Eigenstates.
    line += '{:>6s}{:1s} '.format(entries['eigenstates'], c['eigenstates'])

    # Density.
    line += '{:>5s}{:1s} '.format(entries['density'], c['density'])

    # Custom criteria (optional).
    for name in custom:
        line += f'{entries[name]:>5s}{c[name]:s} '

    # Magnetic moment (optional).
    if ctx.wfs.nspins == 2 or not ctx.wfs.collinear:
        totmom_v, _ = ctx.dens.calculate_magnetic_moments()
        if ctx.wfs.collinear:
            line += f'  {totmom_v[2]:+.4f}'
        else:
            line += ' {:+.1f},{:+.1f},{:+.1f}'.format(*totmom_v)

    log(line.rstrip())
    log.fd.flush()


class SCFEvent:
    """Object to pass the state of the SCF cycle to a convergence-checking
    function."""

    def __init__(self, dens, ham, wfs, niter, log):
        self.dens = dens
        self.ham = ham
        self.wfs = wfs
        self.niter = niter
        self.log = log

    def calculate_forces(self):
        with self.wfs.timer('Forces'):
            F_av = calculate_forces(self.wfs, self.dens, self.ham)
        return F_av


oops = """
Did not converge!

Here are some tips:

1) Make sure the geometry and spin-state is physically sound.
2) Use less aggressive density mixing.
3) Solve the eigenvalue problem more accurately at each scf-step.
4) Use a smoother distribution function for the occupation numbers.
5) Try adding more empty states.
6) Use enough k-points.
7) Don't let your structure optimization algorithm take too large steps.
8) Solve the Poisson equation more accurately.
9) Better initial guess for the wave functions.

See details here:

    https://wiki.fysik.dtu.dk/gpaw/documentation/convergence.html

"""
