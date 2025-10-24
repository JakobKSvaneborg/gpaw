import pytest
from ase.build import molecule

from gpaw.new.etdm.tools import er_localize
from gpaw.new.ase_interface import GPAW

@pytest.fixture(scope="module")
def ibzwfs_from_new_gpaw(tmp_path_factory):
    """Create IBZWaveFunctions using new GPAW."""
    
    atoms = molecule("C6H6")
    atoms.center(vacuum=3.0)
    
    txt_file = tmp_path_factory.mktemp("benzene") / "benzene.txt"
    calc = GPAW(
        mode='pw',
        xc='LDA', 
        nbands=15,
        kpts=(1, 1, 1),
        txt=str(txt_file),
    )
    atoms.calc = calc
    atoms.get_potential_energy()
   
    ibzwfs = calc.dft.ibzwfs
 
    return ibzwfs

def test_er_localize_reproducibility(ibzwfs_from_new_gpaw):
    """Test that ER localization is reproducible."""
    ibzwfs = ibzwfs_from_new_gpaw

    seed = 42
    
    _ = er_localize(ibzwfs, gtol=1e-3, seed=seed)
    
    assert _.energy == pytest.approx(-3.641042186008981, abs=5.e-3)
