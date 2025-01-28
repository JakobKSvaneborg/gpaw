def workflow():
    from myqueue.workflow import run
    with run(function=init):
        run(script='work_function.py')


def init():
    import runpy
    from pathlib import Path
    source = Path('Al100.py').read_text()
    source = source.replace('k = ...', 'k = 6')
    source = source.replace('N = ...', 'N = 5')
    path = Path('al10065.py')
    path.write_text(source)
    runpy.run_path(str(path))
