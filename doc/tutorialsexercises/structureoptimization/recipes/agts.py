def workflow():
    from myqueue.workflow import run
    run(script='fixcell_relax.py')
    run(script='full_relax.py')
