# web-page: pt-atom.csv
from myqueue.workflow import run


def workflow():
    with run(script='pt.py'):
        run(script='projections.py')
