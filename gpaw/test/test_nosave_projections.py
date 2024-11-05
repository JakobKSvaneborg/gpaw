from ase.build import bulk
from gpaw.new.ase_interface import GPAW


def test_no_save_projections(in_tmp_dir):
    atoms = bulk('Si')
    atoms.calc = GPAW(mode='pw', kpts=[2, 2, 2], txt=None)
    atoms.get_potential_energy()
    atoms.calc.write('gs_noprojs.gpw', include_projections=False)

    newcalc = GPAW('gs_noprojs.gpw')
    for wfs in newcalc.dft.ibzwfs:
        assert wfs._P_ani is None
