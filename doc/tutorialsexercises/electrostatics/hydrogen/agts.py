# web-page: h.png
def workflow():
    from myqueue.workflow import run
    with run(script='h.py'):
        run(function=plot)


def plot():
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.io import read
    with open('h.py') as fd:
        code = fd.read().replace('ae', 'paw')
    exec(code)
    ecut = range(200, 1001, 100)
    ae = np.array([read(f'H-{e}-ae.txt').get_potential_energy()
                   for e in ecut])
    paw = np.array([read(f'H-{e}-paw.txt').get_potential_energy()
                    for e in ecut])
    plt.figure(figsize=(6, 4))
    plt.plot(ecut[:-1], ae[:-1] - ae[-1], label='ae')
    plt.plot(ecut[:-1], paw[:-1] - paw[-1], label='paw')
    plt.legend(loc='best')
    plt.xlabel('ecut [eV]')
    plt.ylabel('E(ecut)-E(1000 eV)')
    plt.savefig('h.png')
