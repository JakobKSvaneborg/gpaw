"""GPAW's log-file parser.

Example:

>>> d = parse('''
... a: 117
... b: hello
...
... Indented block:
...   a: (1, 2)
...
... Table
... |ignored|header|  |
... |-------|------|--|
... |a      |1     |2 |
... |b      |3     |4 |
...
... ''')
>>> for k, v in d[0].items():
...     print(k, v)
a 117
b hello
indented_block {'a': [1, 2]}
table [['a', 1, 2], ['b', 3, 4]]
"""
import re
from typing import Any, Iterable, Generator, Iterator


def parse(lines: Iterable[str],
          keys: set[str] | None = None) -> list[dict]:
    """Parse GPAW's log-file to list of dicts.

    One dict for each atomic configuration.
    """
    if isinstance(lines, str):
        lines = lines.splitlines()
    dicts: list[dict] = []
    _parse(line_iter(lines), dicts=dicts, keys=keys)
    return dicts


def normalize_key(key: str) -> str:
    return key.lower().replace(' ', '_').replace('-', '_')


class ConvergedFloat(float):
    """Special float like "-8.4c" ("c" for converged)."""


def line_iter(lines: Iterable[str]) -> Generator[str, None, None]:
    """Yield lines from line-generator.

    Lines are stripped for leading and trailing spaces.
    Special 'DEDENT' lines are yielded everytime an
    indented block of lines stops.  Indented blocks begin
    after lines ending with a colon.
    """
    indent = 0
    for rawline in lines:
        line = rawline.lstrip()
        if line:
            i = len(rawline) - len(line)
            if i > indent:
                continue
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


def table(lines: list[str]) -> list[list]:
    """Parse table lines.

    >>> table(['|1|1.1 |a|',
    ...        '|2|2.2 |b|',
    ...        '|  ...   |',
    ...        '|9|9.01|c|'])
    [[1, 1.1, 'a'], [2, 2.2, 'b'], [9, 9.01, 'c']]
    """
    rows = []
    for line in lines:
        parts = line[1:-1].split('|')
        if len(parts) == 1 and parts[0].strip() == '...':
            continue
        rows.append([parse_str(x.strip()) for x in parts])
    return rows


def _parse(lines: Iterator[str],
           indents: int = 0,
           dicts: list[dict] | None = None,
           keys: set[str] | None = None) -> dict:
    dct: dict[str, Any] = {}
    key = None
    while (line := next(lines)) != 'DEDENT':
        value: Any
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
            key, _, val = line.partition(':')
            key = normalize_key(key)
            if keys is None or key in keys:
                value = parse_str(val.strip())
            else:
                # skip parsing:
                key = None
                continue
        elif line.startswith('|') and key is not None:
            # table:
            key = normalize_key(key)
            while not next(lines).startswith('|-'):
                pass
            rows = []
            while (line := next(lines)).startswith('|'):
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


def main():
    import argparse
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('key', nargs='*')
    args = parser.parse_args()
    with open(args.filename) as fd:
        blocks = parse(fd, keys=set(args.key) if args.key else None)
    print('Blocks:', len(blocks))
    for n, dct in enumerate(blocks):
        print(n, '=' * 70)
        pprint.pp(dct)


def h2():
    import io
    from gpaw.dft import GPAW
    from ase import Atoms
    h = Atoms('H', cell=[2, 2, 2], pbc=1, magmoms=[1])
    txt = io.StringIO()
    h.calc = GPAW(mode='pw', txt=txt)
    h.get_potential_energy()
    h.get_forces()
    h.get_stress()
    out = txt.getvalue()
    print(out)
    ds = parse(iter(out.splitlines()), keys={'atoms', 'unit_cell'})
    print(ds)
    ds = parse(iter(out.splitlines()))
    import pprint
    pprint.pp(ds)


if __name__ == '__main__':
    h2()#main()
