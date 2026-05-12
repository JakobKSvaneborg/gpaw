from myqueue.workflow import run


def workflow():
    run(script='benzene-dimer-T-shaped.py', cores=96, tmax='20h')
    run(script='adenine-thymine_complex_stack.py', cores=4, tmax='2h')
    run(script='graphene_hirshfeld.py')
