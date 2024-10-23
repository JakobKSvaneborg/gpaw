# creates: intro/intro.ipynb
# creates: batteries/batteries1.ipynb
# creates: batteries/batteries2.ipynb
# creates: batteries/batteries3.ipynb
# creates: catalysis/n2_on_metal.ipynb, catalysis/neb.ipynb
# creates: catalysis/vibrations.ipynb, catalysis/convergence.ipynb
# creates: magnetism/magnetism1.ipynb, magnetism/magnetism2.ipynb
# creates: magnetism/magnetism3.ipynb,
# creates: machinelearning/machinelearning.ipynb
# creates: excitedstates/es1.ipynb, excitedstates/es2.ipynb
# creates: excitedstates/es3.ipynb, excitedstates/es4.ipynb
from pathlib import Path
from gpaw.utilities.nbrun import py2ipynb
import sys

teacher = False
if len(sys.argv) == 2:
    if sys.argv[1] == '--teacher':
        teacher = True
    else:
        raise ValueError(f'Unknown command line argument: {sys.argv[1]}')

kernel = {'display_name': 'CAMD2024',
          'language': 'python',
          'name': 'camd2024'}

for path in Path().glob('*/*.py'):
    if path.read_text().startswith('# %%\n'):
        print(path)
        py2ipynb(path, kernel=kernel, teachermode=teacher)
