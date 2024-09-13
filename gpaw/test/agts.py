from myqueue.workflow import run


def workflow():
    run(module='pytest', args=['-mslow'], cores=2)
