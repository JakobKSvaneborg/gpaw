# creates: ugd.csv, uga.csv, pwd.csv, pwa.csv, m.csv
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.matrix import Matrix

for cls in [UGDesc,
            PWDesc,
            UGArray,
            PWArray,
            Matrix]:
    name = ''.join(x for x in cls.__name__ if x.isupper()).lower()
    mod = cls.__module__
    mod = mod.replace('.plane_waves', '')
    mod = mod.replace('.uniform_grid', '')
    print(name, mod)
    with open(f'{name}.csv', 'w') as fd:
        for name, meth in cls.__dict__.items():
            if name[0] != '_':
                try:
                    print(name)
                    doc = meth.__doc__.splitlines()[0]
                except AttributeError:
                    doc = '...'
                print(f':meth:`~{mod}.{cls.__name__}.{name}`, "{doc}"',
                      file=fd)
