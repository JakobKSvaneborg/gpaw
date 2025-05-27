from ase.build import mx2
from gpaw.new.ase_interface import GPAW


def mos2wds(shifts: list[tuple[float, float, int]], tag: str) -> None:
    """WS2 layer on top of MoS2 layer."""
    atoms = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.13,
                size=(1, 1, 1))
    atoms += mx2(formula='WS2', kind='2H', a=3.184, thickness=3.15,
                 size=(1, 1, 1))
    atoms.positions[3:, 2] += 3.6 + (3.13 + 3.15) / 2
    atoms.positions[3:] += [-1 / 3, 1 / 3, 0] @ atoms.cell
    atoms.center(vacuum=6.0, axis=2)
    k = 6
    atoms.calc = GPAW(mode='lcao',
                      basis='dzp',
                      nbands='nao',
                      kpts=dict(size=(k, k, 1), gamma=True),
                      eigensolver={'name': 'scissors',
                                   'shifts': shifts},
                      txt=f'{tag}.txt')
    atoms.get_potential_energy()
    bp = atoms.cell.bandpath('GMKG', npoints=50)
    bs_calc = atoms.calc.fixed_density(kpts=bp, symmetry='off')
    bs_calc.write(f'{tag}.gpw')
    bs = bs_calc.band_structure()
    bs.write(f'{tag}.json')


if __name__ == '__main__':
    for i, shifts in enumerate([[],
                                [(-0.5, 0.5, 3)],
                                [(0.5, 0.5, 3), (-0.5, -0.5, 3)]]):
        mos2wds(shifts, f'mos2ws2-{i}')
