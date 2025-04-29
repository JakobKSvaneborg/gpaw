from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters

# Generate setups with 0.5, 1.0, 0.0 core holes in 1s
elements = ['O', 'C', 'N']
coreholes = [0.5, 1.0, 0.0]
names = ['hch1s', 'fch1s', 'xes1s']
functionals = ['LDA', 'PBE']

for el in elements:
    for name, ch in zip(names, coreholes):
        for funct in functionals:
            g = Generator(el, scalarrel=True, xcname=funct,
                          corehole=(1, 0, ch), nofiles=True)
            g.run(name=name, **parameters[el])

# Generate setups with 0.5, 1.0 core holes in 2p
el = 'S'
coreholes = [0.5, 1.0]
names = ['hch2p', 'fch2p']

for name, ch in zip(names, coreholes):
    for funct in functionals:
        g = Generator(el, scalarrel=True, xcname=funct,
                      corehole=(2, 1, ch), nofiles=True)
        g.run(name=name, **parameters[el])
