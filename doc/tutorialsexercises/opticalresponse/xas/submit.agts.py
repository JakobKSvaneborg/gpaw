# web-page: xas_h2o_spectrum.png, h2o_xas_box.png

def check():
    exec(open('plot.py').read())
    e_dks = float(open('dks.result').readline().split()[2])
    assert abs(e_dks - 532.502) < 0.001
    exec(open('h2o_xas_box2.py').read())


def workflow():
    from myqueue.workflow import run
    with run(script='setups.py'):
        r1 = run(script='run.py', cores=8, tmax='25m')
        r2 = run(script='dks.py', cores=8, tmax='25m')
        r3 = run(script='h2o_xas_box1.py', cores=8, tmax='25m')
        with run(script='diamond1.py', cores=8):
            run(script='diamond2.py', cores=8, tmax='1d')
    run(function=check, deps=[r1, r2, r3])
