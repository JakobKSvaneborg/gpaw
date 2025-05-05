from pathlib import Path

from myqueue.workflow import run


def workflow():
    run(script='plot_overlap_projections.py')
    with run(script='mom_h2o.py', cores=8):
        run(function=check_h2o)
    with run(script='domom_co.py', cores=8):
        run(function=check_co)


def check_h2o():
    text = Path('h2o_energies.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Excitation energy triplet'):
            et = float(line.split()[-2])
        elif line.startswith('Excitation energy singlet'):
            es = float(line.split()[-2])
    assert abs(et - 9.21) < 0.005
    assert abs(es - 9.68) < 0.005


def check_co():
    for tag in ['spinpol', 'spinpaired']:
        co = read('co_' + tag + '.txt')
        assert abs(co.get_distance(0, 1) - 1.248) < 0.01
