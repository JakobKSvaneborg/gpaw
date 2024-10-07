import taskblaster as tb
from pathlib import Path
import runpy

# "Import" stuff from the tasks module:
tasks_path = Path(__file__).parent / 'tasks.py'
globals().update(runpy.run_path(tasks_path))



def workflow(runner):
    from ase.build import bulk
    wf = MaterialsWorkflow(
        atoms=bulk('Si'),
        calculator={'mode': 'pw',
                    'kpts': (4, 4, 4),
                    'txt': 'gpaw.txt'})
    runner.run_workflow(wf)
# end-workflow-function-snippet
