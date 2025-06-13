import numpy as np
import pytest

seed = 42


def test_pwlfc_expand():
    from gpaw.new.c import pwlfc_expand as c_call
    from gpaw.purepython import pwlfc_expand as pp_call
    assert pp_call is not c_call

    cc = False
    dtype = np.complex128

    rng = np.random.RandomState(seed)
    GN = 3000
    aN = 5
    sN = 7
    LN = (sN+1)**2

    f_Gs = rng.randn(GN, sN)
    Gk_Gv = rng.randn(GN, 3)
    pos_av = rng.randn(aN, 3)
    eikR_a = rng.randn(aN) \
        + 1j * rng.randn(aN)
    Y_GL = rng.randn(GN, LN)

    gN = GN if np.issubdtype(dtype, np.complexfloating) else 2 * GN
    l_s = np.arange(sN, dtype=np.int32)
    a_J = []
    s_J = []

    for a in range(aN):
        for s in range(sN):
            a_J.append(a)
            s_J.append(s)
    JN = len(a_J)
    a_J = np.asarray(a_J, dtype=np.int32)
    s_J = np.asarray(s_J, dtype=np.int32)

    I_J = np.zeros(JN, dtype=np.int32)
    I1 = 0
    for J, (a, s) in enumerate(zip(a_J, s_J)):
        l = l_s[s]
        I2 = I1 + 2 * l + 1
        I_J[J] = I1
        I1 = I2
    IN = I2

    f_c_GI = np.zeros((gN, IN), dtype=dtype)
    f_pp_GI = np.zeros((gN, IN), dtype=dtype)

    c_call(f_Gs, Gk_Gv, pos_av,
           eikR_a, Y_GL,
           l_s, a_J, s_J,
           cc, f_c_GI)
    pp_call(f_Gs, Gk_Gv, pos_av,
            eikR_a, Y_GL,
            l_s, a_J, s_J,
            cc, f_pp_GI)

    assert f_c_GI == pytest.approx(f_pp_GI, abs=1e-6)
