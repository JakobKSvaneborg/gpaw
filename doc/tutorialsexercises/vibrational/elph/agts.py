from myqueue.workflow import run


def workflow():
    el = run(script='effective_potential.py', cores=16, tmax='4h')
    scf = run(script='scf.py', cores=16, tmax='10m')
    with el:
        sc = run(script='supercell.py', cores=4, tmax='1h')
    with el, sc, scf:
        with run(script='gmatrix.py', cores=1):
            run(function=check)


def check():
    """Read result and make sure it's OK."""
    import numpy as np

    g_sqklnn = np.load("gsqklnn.npy")
    # Deformation potential at VBM, Gamma, LO phonons is
    # about 3.6 eV/A. 3.47 eV/A with these parameters

    g_lnn = g_sqklnn[0, 0, 665, 3:6, 1:4, 1:4]
    # See PHYSICAL REVIEW B 104, 195201 (2021) Eq 12
    M = np.sqrt(np.sum(g_lnn**2))
    assert np.isclose(M, 3.3310379, rtol=1e-3)
