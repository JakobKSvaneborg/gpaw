## Summary

`gpaw.lcao.tools.get_mulliken` computes Mulliken populations as
`np.dot(rho_MM, S_MM).diagonal()`. This only equals the correct
Mulliken population when `S_MM` is real-symmetric (i.e. at the Γ
point, or in calculations without complex k-point phases). For
generic complex k-points the formula pairs indices on `S` the wrong
way round, and the returned charges do not even sum to the correct
number of electrons per state.

A quick symptom: for a non-Γ, non-spin-polarized calculation, Mulliken
weights per band may sum to anything from ~0.4 to ~9 instead of 1.

## Location

`gpaw/lcao/tools.py:44-61`:

```python
def get_mulliken(calc, a_list):
    """Mulliken charges from a list of atom indices (a_list)."""
    Q_a = {}
    for a in a_list:
        Q_a[a] = 0.0
    for kpt in calc.wfs.kpt_u:
        S_MM = calc.wfs.S_qMM[kpt.q]
        nao = S_MM.shape[0]
        rho_MM = np.empty((nao, nao), calc.wfs.dtype)
        calc.wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, rho_MM)
        Q_M = np.dot(rho_MM, S_MM).diagonal()   # <-- wrong for complex S
        ...
```

The same bug appears anywhere `Re((C @ S) * C.conj())` or equivalently
`diag(rho @ S)` is used to partition a wavefunction norm across LCAO
basis functions.

## Reproducer

A small LCAO calculation at a generic (complex) k-point. The script
explicitly calls `get_mulliken` on every atom and compares the sum of
the returned charges to the actual number of valence electrons in the
system. With the current implementation the two disagree
substantially; with the correct contraction they agree to machine
precision.

```python
# mulliken_normalization_test.py
from ase.build import mx2
from gpaw import GPAW
from gpaw.lcao.tools import get_mulliken

atoms = mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, vacuum=5.0)
atoms.calc = GPAW(
    mode='lcao',
    basis='dzp',
    xc='PBE',
    kpts={'size': (3, 3, 1), 'gamma': False},  # shifted off Gamma
    symmetry='off',
    convergence={'density': 1e-3},
    txt='mulliken_test.txt',
)
atoms.get_potential_energy()

# Explicitly call the (buggy) get_mulliken on every atom.
Q_a = get_mulliken(atoms.calc, list(range(len(atoms))))
total_charge = sum(Q_a.values())
n_electrons = atoms.calc.get_number_of_electrons()

print('Mulliken charges per atom:')
for a, Q in Q_a.items():
    print(f'  atom {a} ({atoms[a].symbol}): {Q:.4f}')

print(f'\nSum of Mulliken charges:        {total_charge:.4f}')
print(f'Expected number of electrons:   {n_electrons:.4f}')
print(f'Discrepancy:                    {total_charge - n_electrons:.4f}')
```

Expected output (abridged): the sum of the returned Mulliken charges
does **not** match `get_number_of_electrons()` — typical discrepancies
are large (factors of two or more), not a rounding-level effect.
Only at a pure Γ-point grid (where `S_MM` happens to be real) does the
sum agree.

## Proposed fix

Replace the Mulliken contraction inside `get_mulliken`. Either:

```python
Q_M = np.dot(rho_MM, S_MM.T).diagonal()
```

or (equivalent for Hermitian S, slightly clearer intent):

```python
Q_M = np.dot(rho_MM, S_MM.conj()).diagonal()
```

A corresponding fix is needed anywhere the pattern
`Re((C @ S) * C.conj())` or `diag(rho @ S)` appears for per-basis-function
populations at complex k.

## Impact

Anywhere `get_mulliken` is called on a calculation with more than the
Γ-point — or on any spin-polarized / bilayer / transport-style LCAO
setup that touches complex k — the returned atomic charges are
quantitatively wrong (not just off by rounding). The sum over all
basis functions on all atoms does not equal the number of electrons
per band.

I hit this while building a layer-projected band structure plot for
a TMD bilayer; only the Γ-point rank printed `sum=1.000000`, every
other k-point was off by factors of 2–9.
