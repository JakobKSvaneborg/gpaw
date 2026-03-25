"""GPAW's log-file parser.

Example:

>>> parse('''
a: 117
b: hello

Indented block:
    a: (1, 2)

table
--------------
ignored header
--------------
a      1     2
b      3     4
--------------
''')

"""
import re
from typing import Any, Iterable, Generator


def parse(lines: Iterable[str],
          keys: set[str] | None = None) -> list[dict]:
    """Parse GPAW's log-file to list of dicts.

    One dict for each atomic configuration.
    """
    dicts = []
    _parse(line_iter(lines), dicts=dicts, keys=keys)
    return dicts


class ConvergedFloat(float):
    """Special float like "-8.4c" ("c" for converged)."""


def parse_str(s: str) -> Any:
    """Convert str to int, float or str or list any of those or ...

    >>> parse_str('1.2')
    1.2
    >>> parse_str('AB')
    'AB'
    >>> parse_str('( 1, 1.2, AB)')
    [1, 1.2, 'AB']
    """
    if s == '-':
        return float('nan')
    if ',' in s:
        s = re.sub(r',\s+', ',', s)
        s = re.sub(r'\(\s+', '(', s)
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


def table(lines: list[str]) -> list[list] | dict[int, list]:
    """Parse table lines.

    >>> table(['1 1.1 a', '2 2.2 b', '3 3.3 c'])
    [[1, 1.1, 'a'], [2, 2.2, 'b'], [3, 3.3, 'c']]
    >>> table(['1 1.1 a', '...', '3 3.3 c'])
    {1: [1.1, 'a'], 3: [3.3, 'c']}
    """
    rows = []
    missing_lines = False
    for line in lines:
        if line == '...':
            missing_lines = True
        else:
            line = re.sub(r',\s+', ',', line)
            line = re.sub(r'\(\s+', '(', line)
            rows.append([parse_str(x) for x in line.split()])
    if missing_lines:
        dct = []
        for key, *row in rows:
            dct[key] = row
        return dct
    return rows


def line_iter(lines: Iterable[str]) -> Generator[str]:
    """Yield lines from line-generator.

    Lines are stripped for leading and trailing spaces.
    Special 'DEDENT' lines are yielded everytime and
    indented block of lines stops.  Indented blocks begin
    after lines ending with a colon.
    """
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


def _parse(lines: Iterable[str],
           indents: int = 0,
           dicts: list[dict] | None = None,
           keys: set[str] | None = None) -> dict:
    dct = {}
    key = None
    while (line := next(lines)) != 'DEDENT':
        if line.endswith(':'):
            # indented block:
            key = normalize_key(line[:-1])
            if keys is None or key in keys:
                value = _parse(lines, indents + 1)
            else:
                # skip parsing:
                while next(lines) != 'DEDENT':
                    pass
                key = None
                continue
        elif ':' in line:
            # "key: value" pair:
            key, _, value = line.partition(':')
            key = normalize_key(key)
            if keys is None or key in keys:
                value = parse_str(value.strip())
            else:
                # skip parsing:
                key = None
                continue
        elif line.startswith('--') and key is not None:
            # table:
            key = normalize_key(key)
            while not next(lines).startswith('--'):
                pass
            rows = []
            while not (line := next(lines)).startswith('--'):
                rows.append(line)
            if keys is None or key in keys:
                value = table(rows)
            else:
                # skip parsing:
                key = None
                continue
        else:
            key = line
            continue
        if dicts is not None and key in dct:
            dicts.append(dct)
            dct = {}
        if re.match('[a-zA-Z_]+', key):
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
    ds = parse(iter(out.splitlines()), keys={'atoms', 'unit_cell'})
    print(ds)
    parse(iter(out.splitlines()))


if __name__ == '__main__':
    h2()
