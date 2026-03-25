r"""
  __  _  _
 | _ |_)|_||  |
 |__||  | ||/\| - 26.4.0

 dsfg
 asdg
 sadg

user: jensj
x:    42

Species:
  H:
    a: ...
  Li:
    a: asfag

Atoms:
  positions
  -------------
   symbol x y z magmom
  --------------------------
  0 H  1.2 2.0 0.0
  1 Li 1.2 2.0 0.0
  --------------

cell:
  axes
  ---
   x   y   z
  ----------
  1 2 3
  2 3 4
  4 5 6
  ---------
  lengths: 1, 2, 3

SCF:
  iterations
  -------------
  dafg df dfag
  ----------------
  iter 0 12:03:47 1.4  -
  iter 1 11:03:48 1.4c -9.5
  ----------------

  steps: 2

Atoms: 54
"""
import re


class ConvergedFloat(float):
    pass


def parse_str(s):
    if s == '-':
        return float('nan')
    if ',' in s:
        s = re.sub(r',\s+', ',', s)
        if s.startswith('(('):
            return [parse_str(x) for x in s[2:-2].split('),(')]
        if s.startswith('('):
            return [parse_str(x) for x in s[1:-1].split(',')]
        return [parse_str(x) for x in s.split(',')]
    if s.endswith('c'):
        try:
            x = float(s[:-1])
        except ValueError:
            return s
        return ConvergedFloat(x)
    if len(s) == 8 and ':' in s:
        ho, mi, se = (int(x) for x in s.split(':'))
        return (ho * 60 + mi) * 60 + se
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def table(lines):
    rows = []
    missing_lines = False
    for line in lines:
        if line == '...':
            missing_lines = True
        else:
            line = re.sub(r',\s+', ',', line)
            rows.append([parse_str(x) for x in line.split()])
    if missing_lines:
        dct = []
        for key, *row in rows:
            dct[key] = row
        return dct
    return rows


def line_iter(lines):
    indent = 0
    for rawline in lines:
        print(rawline)
        line = rawline.lstrip()
        if line:
            i = len(rawline) - len(line)
            while i < indent:
                yield 'DEDENT'
                indent -= 2
            line = line.split('#')[0].rstrip()
            if line.endswith(':'):
                indent += 2
            yield line
    while indent >= 0:
        yield 'DEDENT'
        indent -= 2


def parse(lines, indents=0, dicts=None, keys=None):
    dct = {}
    key = None
    while (line := next(lines)) != 'DEDENT':
        if line.endswith(':'):
            key = normalize_key(line[:-1])
            if keys is None or key in keys:
                value = parse(lines, indents + 1)
            else:
                while next(lines) != 'DEDENT':
                    pass
                key = None
                continue
        elif ':' in line:
            key, _, value = line.partition(':')
            key = normalize_key(key)
            if keys is None or key in keys:
                value = parse_str(value.strip())
            else:
                key = None
                continue
        elif line.startswith('--') and key is not None:
            key = normalize_key(key)
            while not next(lines).startswith('--'):
                pass
            rows = []
            while not (line := next(lines)).startswith('--'):
                rows.append(line)
            if keys is None or key in keys:
                value = table(rows)
            else:
                key = None
                continue
        else:
            key = line
            continue
        if dicts is not None and key in dct:
            dicts.append(dct)
            dct = {}
        if not re.match('[a-zA-Z_ ]+', key):
            raise ValueError(f'Bad key: {key}')
        dct[key] = value
        key = None
    if dicts is not None:
        dicts.append(dct)
    return dct


def normalize_key(key):
    return key.lower().replace(' ', '_')


def h2():
    import io
    from gpaw.dft import GPAW
    from ase import Atoms
    h = Atoms('H', cell=[2, 2, 2], pbc=1)
    txt = io.StringIO()
    h.calc = GPAW(mode='pw', txt=txt)
    h.get_potential_energy()
    out = txt.getvalue()
    ds = []
    parse(line_iter(iter(out.splitlines())), dicts=ds)
    ds = []
    parse(line_iter(iter(out.splitlines())), dicts=ds, keys={'atoms', 'unit_cell'})
    print(ds)


if __name__ == '__main__':
    h2()
    if 0:
        ds = []
        parse(line_iter(iter(__doc__.splitlines())), ds)
        print(ds)
