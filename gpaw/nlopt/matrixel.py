from os import path

import numpy as np
from ase.parallel import parprint
from ase.units import Ha
from ase.utils.timing import Timer

from gpaw.new.ase_interface import GPAW
from gpaw.fd_operators import Gradient
from gpaw.mpi import world, serial_comm
from gpaw.utilities.progressbar import ProgressBar


class WaveFunctionAdapter:
    def __init__(self, calc):
        # We don't know why this is needed
        if calc.parameters.mode['name'] == 'lcao':
            calc.initialize_positions(calc.atoms)
        self.ibzk_kc = calc.get_ibz_k_points()
        self.nb_full = calc.get_number_of_bands()

        state = calc.calculation.state
        self.grid = state.density.nt_sR.desc.new(dtype=complex)
        self.ns = state.density.ndensities
        self.ibzwfs = state.ibzwfs

        wfs = calc.wfs
        self.gd = wfs.gd
        self.nabla_aiiv = [setup.nabla_iiv for setup in wfs.setups]

    def get_pseudo_wave_function(self, ni, nf, k_ind, spin):
        psit_nX = self.ibzwfs.wfs_qs[k_ind][spin].psit_nX[ni:nf]
        return psit_nX.ifft(grid=self.grid, periodic=True).data

    def get_wave_function_projections(self, ni, nf, k_ind, spin):
        P_ani = {a: P_ni[ni:nf] for a, P_ni in
                 self.ibzwfs.wfs_qs[k_ind][spin].P_ani.items()}
        return P_ani


def get_mml(gs, spin=0, ni=None, nf=None, timer=None):
    """Compute the momentum matrix elements.

    Input:
        gs_name         Ground state file name
        spin            Which spin channel (for spin-polarized systems 0 or 1)
        ni, nf          First and last band to compute the mml (0 to nb)
        timer           Timer to keep track of time
    Output:
        p_kvnn2         A big array in master
    """

    # Start the timer
    if timer is None:
        timer = Timer()
    parprint('Calculating momentum matrix elements...')

    # Specify desired range and number of bands in calculation
    nb_full = gs.nb_full
    ni = int(ni) if ni is not None else 0
    nf = int(nf) if nf is not None else nb_full
    nf = nb_full + nf if (nf < 0) else nf
    nb = nf - ni

    # Spin input
    assert spin < gs.ns, 'Wrong spin input'

    # Real and reciprocal space parameters
    icell_cv = (2 * np.pi) * gs.grid.icell
    nk = np.shape(gs.ibzk_kc)[0]

    # Parallelisation and memory estimate
    rank = world.rank
    size = world.size
    nkcore = int(np.ceil(nk / size))  # Number of k-points pr. core
    est_mem = 2 * 3 * nk * nb**2 * 16 / 2**20
    parprint(f'At least {est_mem:.2f} MB of memory is required.')

    # Allocate the matrix elements
    p_kvnn = np.zeros((nkcore, 3, nb, nb), dtype=complex)

    # if calc.parameters['mode'] == 'lcao':
    nabla_v = [
        Gradient(
            gs.gd, vv, 1.0, 4,
            complex).apply for vv in range(3)]
    phases = np.ones((3, 2), dtype=complex)

    # Initial call to print 0% progress
    ik = 0
    if rank == 0:
        pb = ProgressBar()

    # Calculate matrix elements in loop over k-points
    for ik in range(nkcore):
        k_ind = rank + size * ik
        if k_ind >= nk:
            break
        k_c = gs.ibzk_kc[k_ind]
        k_v = np.dot(k_c, icell_cv)

        # Get the wavefunctions
        with timer('Get wavefunctions and projections'):
            u_nR = gs.get_pseudo_wave_function(ni=ni, nf=nf,
                                               k_ind=k_ind, spin=spin)
            P_ani = gs.get_wave_function_projections(ni=ni, nf=nf,
                                                     k_ind=k_ind, spin=spin)

        # Now compute the momentum part
        grad_nv = gs.gd.zeros((nb, 3), complex)
        with timer('Momentum calculation'):
            # Get the derivative
            for iv in range(3):
                for ib in range(nb):
                    nabla_v[iv](u_nR[ib], grad_nv[ib, iv], phases)

            # Compute the integral
            p_vnn = np.transpose(
                gs.gd.integrate(u_nR, grad_nv), (2, 0, 1))

            # Add the overlap part
            M_nn = np.array([gs.gd.integrate(
                u_nR[ib], u_nR) for ib in range(nb)])
            for iv in range(3):
                p_vnn[iv] += 1j * k_v[iv] * M_nn

        # The PAW corrections are added
        with timer('Add the PAW correction'):
            for ia, nabla_iiv in enumerate(gs.nabla_aiiv):
                P0_ni = P_ani[ia]

                # Loop over components
                for iv in range(3):
                    nabla_ii = nabla_iiv[:, :, iv]
                    p_vnn[iv] += np.dot(
                        np.dot(P0_ni.conj(), nabla_ii), P0_ni.T)

        # Make it momentum and store it
        p_kvnn[ik] = -1j * p_vnn

        # Print the progress
        if rank == 0:
            pb.update(ik / nkcore)
        ik += 1

    if rank == 0:
        pb.finish()

    # Gather all data to the master
    p_kvnn2 = []
    with timer('Gather the data to master'):
        parprint('Gathering date to the master.')
        recv_buf = None
        if rank == 0:
            recv_buf = np.empty((size, nkcore, 3, nb, nb),
                                dtype=complex)
        world.gather(p_kvnn, 0, recv_buf)
        if rank == 0:
            p_kvnn2 = np.zeros((nk, 3, nb, nb), dtype=complex)
            for ii in range(size):
                k_inds = range(ii, nk, size)
                p_kvnn2[k_inds] = recv_buf[ii, :len(k_inds)]

    # Print the timing
    if rank == 0:
        timer.write()

    return p_kvnn2


def make_nlodata(gs_name: str = 'gs.gpw',
                 out_name: str = 'mml.npz',
                 spin: str = 'all',
                 ni: int = 0,
                 nf: int = 0) -> None:

    """Get all required NLO data and store it in a file.

    Writes NLO data to file: w_sk, f_skn, E_skn, p_skvnn.

    Parameters:

    gs_name:
        Ground state file name
    out_name:
        Output filename
    spin:
        Which spin channel ('all', 's0' , 's1')
    ni:
        First band to compute the mml.
    nf:
        Last band to compute the mml (relative to number of bands
        for nf <= 0).

    """

    assert path.exists(gs_name), \
        f'The gs file: {gs_name} does not exist!'
    calc = GPAW(gs_name, txt=None, communicator=serial_comm)

    assert not calc.symmetry.point_group, \
        'Point group symmtery should be off.'

    gs = WaveFunctionAdapter(calc)
    ns = gs.ns
    if spin == 'all':
        spins = list(range(ns))
    elif spin == 's0':
        spins = [0]
    elif spin == 's1':
        spins = [1]
        assert spins[0] < ns, 'Wrong spin input'
    else:
        raise NotImplementedError

    if nf <= 0:
        nf += gs.nb_full

    return _make_nlodata(gs=gs, out_name=out_name,
                         spins=spins, ni=ni, nf=nf)


def _make_nlodata(gs,
                  out_name: str,
                  spins: list,
                  ni: int,
                  nf: int) -> None:

    # Start the timer
    timer = Timer()

    # Get the energy and fermi levels (data is only in master)
    with timer('Get energies and fermi levels'):
        ibzwfs = gs.ibzwfs
        if world.rank == 0:
            # Get the data
            E_skn, f_skn = ibzwfs.get_all_eigs_and_occs()
            # Energy is returned in Ha. For now we will change
            # it to eV avoid altering the module too much.
            E_skn *= Ha

            w_sk = np.array([ibzwfs.ibz.weight_k for s1 in spins])
            bz_vol = np.linalg.det(2 * np.pi * gs.grid.icell)
            w_sk *= bz_vol * ibzwfs.spin_degeneracy

    # Compute the momentum matrix elements
    with timer('Compute the momentum matrix elements'):
        p_skvnn = []
        for s1 in spins:
            p_kvnn = get_mml(gs=gs, spin=s1,
                             ni=ni, nf=nf, timer=timer)
            p_skvnn.append(p_kvnn)

    # Save the output to the file
    if world.rank == 0:
        np.savez(out_name, w_sk=w_sk, f_skn=f_skn[:, :, ni:nf],
                 E_skn=E_skn[:, :, ni:nf], p_skvnn=np.array(p_skvnn, complex))


def get_rml(E_n, p_vnn, pol_v, Etol=1e-6):
    """
    Compute the position matrix elements

    Input:
        E_n             Energies
        p_vnn           Momentum matrix elements
        pol_v           Tensor element
        Etol            Tol. in energy to consider degeneracy
    Output:
        r_vnn, D_vnn    Position and velocity difference matrix el.
    """

    # Useful variables
    nb = len(E_n)
    r_vnn = np.zeros((3, nb, nb), complex)
    D_vnn = np.zeros((3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    # Loop over components
    for v1 in set(pol_v):
        r_vnn[v1] = p_vnn[v1] / (1j * E_nn)
        r_vnn[v1, zeroind] = 0
        p_n = np.diag(p_vnn[v1])
        D_vnn[v1] = np.tile(p_n[:, None], (1, nb)) - \
            np.tile(p_n[None, :], (nb, 1))

    return r_vnn, D_vnn


def get_derivative(E_n, r_vnn, D_vnn, pol_v, Etol=1e-6):
    """
    Compute the generalized derivative of position matrix elements

    Input:
        E_n             Energies
        r_vnn           Momentum matrix elements
        D_vnn           Velocity difference
        pol_v           Tensor element
        Etol            Tol. in energy to consider degeneracy
    Output:
        rd_vvnn         Generilized derivative of position
    """

    # Useful variables
    nb = len(E_n)
    rd_vvnn = np.zeros((3, 3, nb, nb), complex)
    E_nn = np.tile(E_n[:, None], (1, nb)) - \
        np.tile(E_n[None, :], (nb, 1))
    zeroind = np.abs(E_nn) < Etol
    E_nn[zeroind] = 1
    for v1 in set(pol_v):
        for v2 in set(pol_v):
            tmp = (r_vnn[v1] * np.transpose(D_vnn[v2])
                   + r_vnn[v2] * np.transpose(D_vnn[v1])
                   + 1j * np.dot(r_vnn[v1], r_vnn[v2] * E_nn)
                   - 1j * np.dot(r_vnn[v2] * E_nn, r_vnn[v1])) / E_nn
            tmp[zeroind] = 0
            rd_vvnn[v1, v2] = tmp

    return rd_vvnn
