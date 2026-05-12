from myqueue.workflow import run


def workflow():
    run(script='graphene.py', cores=8, tmax='15m')
