import os

# This is recommended to be set as an environment variable
os.environ['GPAW_TRACE'] = '1'

from gpaw.new.ase_interface import GPAW
from gpaw.new.timer import global_timer
from gpaw.utilities.timing import GPUProfiler
from ase.build import graphene

atoms = graphene(size=(5, 5, 1), vacuum=5)
atoms.set_pbc((True, True, False))
atoms.calc = GPAW(
    mode={"name": "pw", "ecut": 500},
    parallel={"gpu": True},
    convergence={"density": 3, "eigenstates": 3, "energy": 10},
    kpts=(1, 1, 1),
    random=True,
)
with global_timer.context(GPUProfiler("gpu")) as timer:
    atoms.get_potential_energy()
