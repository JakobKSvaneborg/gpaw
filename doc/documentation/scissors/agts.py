# web-page: mos2ws2.png
from myqueue.workflow import run


def workflow():
    with run(script='mos2ws2.py'):
        run(script='plot_bs.py')
