def workflow():
    from myqueue.workflow import run
    runs = [run(script='ferro.py', cores=4, tmax='15m'),
            run(script='anti.py', cores=4, tmax='15m'),
            run(script='non.py', cores=2, tmax='15m')]
    run(script='PBE.py', deps=runs)
    for r, name in zip(runs, ['ferro', 'anti', 'non']):
        run(script='../../electronic/dos/dos.py',
            args=[f'{name}.gpw'], deps=[r])
    run(script='pdos.py', deps=[runs[0]])
    run(script='magnetization.py', deps=[runs[1]])
