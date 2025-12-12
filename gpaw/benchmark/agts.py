from gpaw.benchmark.performance_index import workflow as wf
from pathlib import Path
import os


def workflow():
    from myqueue.workflow import run
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
        handles = wf(skip=['ErGe-2M', 'Fe8O8-3M'])
        os.chdir('..')
        run(function=check, args=[mode], deps=handles)


def check(mode: str) -> None:
    assert True
