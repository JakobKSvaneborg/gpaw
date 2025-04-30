from __future__ import annotations


def mode(value=None):
    if value is None:
        return {'name': value}
    if isinstance(value, str):
        return {'name': value}
    gc = value.pop('gammacentered', False)
    assert not gc
    return value


def parallel(value: dict | None = None):
    known = {'kpt', 'domain', 'band',
             'order',
             'stridebands',
             'augment_grids',
             'sl_auto',
             'sl_default',
             'sl_diagonalize',
             'sl_inverse_cholesky',
             'sl_lcao',
             'sl_lrtddft',
             'use_elpa',
             'elpasolver',
             'buffer_size',
             'gpu'}
    if value is not None:
        if not value.keys() <= known:
            key = (value.keys() - known).pop()
            raise ValueError(
                f'Unknown key: {key!r}. Must be one of {", ".join(known)}')
        return value
    return {}


def symmetry(value='undefined'):
    """Use of symmetry."""
    if value == 'undefined':
        value = {}
    elif value is None or value == 'off':
        value = {'point_group': False, 'time_reversal': False}
    return value
