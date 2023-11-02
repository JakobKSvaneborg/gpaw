import numpy as np
from ase.units import Bohr, _hbar, _e, _me
from ase.utils.timing import Timer
from ase.parallel import parprint
from gpaw.mpi import world
from gpaw.nlopt.matrixel import get_rml, get_derivative
from gpaw.utilities.progressbar import ProgressBar


def get_shift(
        nlodata,
        freqs=[1.0],
        eta=0.05,
        pol='yyy',
        eshift=0.0,
        ftol=1e-4, Etol=1e-6,
        band_n=None,
        out_name='shift.npy'):
    """Calculate RPA shift current for nonmagnetic semiconductors.

    Parameters
    ==========
    nlodata:
        Data object of class NLOData.
    freqs:
        Excitation frequency array (a numpy array or list).
    eta:
        Broadening, a number or an array (default 0.05 eV).
    pol:
        Tensor element (default 'yyy').
    Etol, ftol:
        Tolerance in energy and fermi to consider degeneracy.
    band_n:
        List of bands in the sum (default 0 to nb).
    out_name:
        Output filename (default 'shift.npy').

    Returns
    =======
    np.ndarray
        Numpy array containing the spectrum and frequencies.
    """

    # Start a timer
    timer = Timer()
    parprint(f'Calculating shift current (in {world.size:d} cores).')

    # Useful variables
    pol_v = ['xyz'.index(ii) for ii in pol]
    w_l = np.array(freqs)
    nw = len(freqs)
    parprint(f'Calculation for element {pol}.')

    # Load the required data
    with timer('Load and distribute the data'):
        k_info = nlodata.distribute()
        if k_info:
            tmp = list(k_info.values())[0]
            nb = len(tmp[1])
            nk = len(k_info) * world.size  # Approximately
            if band_n is None:
                band_n = list(range(nb))
            mem = 6 * 3 * nk * nb**2 * 16 / 2**20
            parprint(f'At least {mem:.2f} MB of memory is required.')

    # Initial call to print 0% progress
    count = 0
    ncount = len(k_info)
    if world.rank == 0:
        pb = ProgressBar()

    # Initialize the outputs
    sum2_l = np.zeros((nw), complex)

    # Do the calculations
    for _, (we, f_n, E_n, p_vnn) in k_info.items():
        with timer('Position matrix elements calculation'):
            r_vnn, D_vnn = get_rml(E_n, p_vnn, pol_v, Etol=Etol)

        with timer('Compute generalized derivative'):
            rd_vvnn = get_derivative(E_n, r_vnn, D_vnn, pol_v, Etol=Etol)

        with timer('Sum over bands'):
            tmp = shift_current(
                eta, w_l, f_n, E_n, r_vnn, rd_vvnn, pol_v,
                band_n, ftol, Etol, eshift)

        # Add it to previous with a weight
        sum2_l += tmp * we

        # Print the progress
        if world.rank == 0:
            pb.update(count / ncount)
        count += 1

    if world.rank == 0:
        pb.finish()

    with timer('Gather data from cores'):
        world.sum(sum2_l)

    # Make the output in SI unit
    dim_init = -1j * _e**3 / (2 * _hbar * (2.0 * np.pi)**3)
    dim_sum = (_hbar / (Bohr * 1e-10))**3 / \
        (_e**4 * (Bohr * 1e-10)**3) * (_hbar / _me)**3
    dim_SI = 1j * dim_init * dim_sum  # 1j due to imag in loop
    sigma_l = dim_SI * sum2_l.real

    # A multi-col output
    shift = np.vstack((freqs, sigma_l))

    # Save it to the file
    if world.rank == 0:
        np.save(out_name, shift)

        # Print the timing
        timer.write()

    return shift


def shift_current(
        eta, w_l, f_n, E_n, r_vnn, rd_vvnn, pol_v,
        band_n=None, ftol=1e-4, Etol=1e-6, eshift=0):
    """
    Loop over bands for computing in length gauge

    Input:
        eta             Broadening
        w_l             Complex frequency array
        f_n             Fermi levels
        E_n             Energies
        r_vnn           Momentum matrix elements
        rd_vvnn         Generalized derivative of position
        pol_v           Tensor element
        band_n          Band list
        Etol, ftol      Tol. in energy and fermi to consider degeneracy
        eshift          Bandgap correction
    Output:
        sum2_l          Output array
    """

    # Initialize variable
    nb = len(f_n)
    if band_n is None:
        band_n = list(range(nb))
    sum2_l = np.zeros((w_l.size), complex)

    # Loop over bands
    for nni in band_n:
        for mmi in band_n:
            # Remove the non important term (use TRS)
            if mmi <= nni:
                continue
            fnm = f_n[nni] - f_n[mmi]
            Emn = E_n[mmi] - E_n[nni] + fnm * eshift

            # Two band part
            if np.abs(fnm) > ftol:
                tmp = np.imag(
                    r_vnn[pol_v[1], mmi, nni]
                    * rd_vvnn[pol_v[0], pol_v[2], nni, mmi]
                    + r_vnn[pol_v[2], mmi, nni]
                    * rd_vvnn[pol_v[0], pol_v[1], nni, mmi]) \
                    * (eta / (np.pi * ((w_l - Emn) ** 2 + eta ** 2))
                        - eta / (np.pi * ((w_l + Emn) ** 2 + eta ** 2)))

                sum2_l += fnm * tmp

    return 2 * np.pi * sum2_l
