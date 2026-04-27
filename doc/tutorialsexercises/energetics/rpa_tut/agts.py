from myqueue.workflow import run

ds = [1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
      4.0, 5.0, 6.0, 10.0]


def workflow():
    with run(script='gs_N2.py', cores=16):
        r1 = run(script='frequency.py', tmax='3h')
        r2 = run(script='con_freq.py', cores=2, tmax='16h')
        r3 = run(script='rpa_N2.py', cores=48, tmax='30m')
    with r1, r2:
        run(script='plot_w.py')
    with r2:
        run(script='plot_con_freq.py')
    with r3:
        run(script='extrapolate.py')

    with r3:  # r1, r2, r3:
        run(shell='rm', args=['N.gpw', 'N2.gpw'])  # clean up

    deps = []
    for d in ds:
        r = run(script='c2cu.py', args=['RPA', d], cores=40, tmax='5h',
                weight=1)
        deps.append(r)

    for xc in ['LDA', 'vdW-DF']:
        r = run(script='c2cu.py', args=[xc] + ds, cores=24, tmax='5h')
        deps.append(r)

    run(script='plot_c2cu.py', deps=deps)
