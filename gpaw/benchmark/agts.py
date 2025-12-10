from gpaw.benchmark.performance_index import workflow as wf
from pathlib import Path
import os


def workflow():
    for mode in ['pw', 'lcao', 'fd']:
        dir = Path(mode)
        if not dir.is_dir():
            dir.mkdir()
            if mode == 'lcao':
                (dir / 'params.json').write_text(
                    '{"mode": "lcao", "basis": "dzp", "h": 0.15}\n')
            elif mode == 'fd':
                (dir / 'params.json').write_text(
                    '{"mode": "fd", "h": 0.15}\n')
        os.chdir(dir)
        wf(skip=['ErGe-2M', 'Fe8O8-3M'], mode=mode)
        os.chdir('..')
