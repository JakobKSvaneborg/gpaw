from gpaw.atom.generator import Generator
from gpaw.atom.basis import BasisMaker
from gpaw.atom.configurations import parameters, parameters_extra

atom = 'Ag'
xc = 'GLLBSC'
name = 'my'
if atom in parameters_extra:
    args = parameters_extra[atom]  # Choose the smaller setup
else:
    args = parameters[atom]  # Choose the larger setup
args.update(dict(name=name, exx=True))

# Generate setup
generator = Generator(atom, xc, scalarrel=True)
setup = generator.run(write_xml=True, **args)

# Generate basis
bm = BasisMaker.from_setup_and_generator(
    setup, generator, name=f'{name}.{xc}')
basis = bm.generate(zetacount=2, polarizationcount=0,
                    jvalues=[0, 1, 2])  # include d, s and p
basis.write_xml()
