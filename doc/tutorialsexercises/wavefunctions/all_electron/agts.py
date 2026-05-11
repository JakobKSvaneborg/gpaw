from myqueue.workflow import run


def workflow():
    run(script='NaCl.py', tmax='30m')
    run(script='C6H6.py', tmax='30m')
