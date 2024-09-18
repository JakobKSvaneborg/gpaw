def workflow():
    from myqueue.workflow import run
    run(module='pytest', args=['-m', 'slow'], cores=2)
