"""Code for summerschool notebooks."""
import json
import sys
from pathlib import Path
from typing import Any


def py2ipynb(path: Path,
             kernel: dict[str, str] | None = None,
             teachermode: bool | None = False) -> None:
    """Convert Python script to ipynb file.

    Hides cells marked with "# teacher" and replaces lines marked with
    "# student: ..."::

        answer = 42  # student: answer = ...

    gives::

        answer = ...

    """
    cells = []
    text = path.read_text()
    assert text.startswith('# %%\n')
    chunks = text[5:].split('\n\n# %%\n')

    for chunk in chunks:
        cell_type = 'code'
        if chunk.startswith(('"""', 'r"""')):
            chunk = chunk.strip('r\n')
            chunk = chunk.strip('"')
            cell_type = 'markdown'

        cell: dict[str, Any] = {
            'cell_type': cell_type,
            'metadata': {},
            'source': chunk.splitlines(True)}

        if cell_type == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            lines = cell['source']
            for i, line in enumerate(lines):
                if ' # student:' in line and not teachermode:
                    a, b = (x.strip() for x in line.split('# student:'))
                    lines[i] = line.split(a)[0] + b + '\n'
                elif line.startswith('# magic: '):
                    lines[i] = line[9:]
                elif line.lower().startswith('# teacher') and not teachermode:
                    del lines[i:]
                    break

        cells.append(cell)

    outpath = path.with_suffix('.ipynb')
    if kernel is None:
        kernel = {'display_name': 'Python 3',
                  'language': 'python',
                  'name': 'python3'}
    outpath.write_text(
        json.dumps(
            {'cells': cells,
             'metadata': {
                 'kernelspec': kernel,
                 'language_info': {
                     'codemirror_mode': {'name': 'ipython', 'version': 3},
                     'file_extension': '.py',
                     'mimetype': 'text/x-python',
                     'name': 'python',
                     'nbconvert_exporter': 'python',
                     'pygments_lexer': 'ipython3',
                     'version': '3.6.1'}},
             'nbformat': 4,
             'nbformat_minor': 1},
            indent=2))


def py2rst(path: Path) -> None:
    """Convert Python script to reST file."""
    text = path.read_text()
    assert text.startswith('# %%\n')
    chunks = text[5:].split('\n\n# %%\n')

    output = []
    for chunk in chunks:
        if chunk.startswith(('"""', 'r"""')):
            chunk = chunk.strip('r\n')
            chunk = chunk.strip('"')
            for line in chunk.splitlines():
                if line.startswith('# '):
                    output += [
                        '',
                        '=' * (len(line) - 2), line[2:], '=' * (len(line) - 2)]
                elif line.startswith('## '):
                    output += [
                        '', line[3:], '=' * (len(line) - 3)]
                elif line.startswith('### '):
                    output += [
                        '', line[4:], '-' * (len(line) - 4)]
                else:
                    output.append(line)
        else:
            output += [
                '',
                '.. code::',
                '']
            lines = chunk.splitlines()
            for line in lines:
                if line.startswith('# magic: '):
                    print(line)
                output.append('    ' + line)

        output.append('')

    outpath = path.with_suffix('.rst')
    assert not outpath.is_file(), outpath
    outpath.write_text(
        '\n'.join(output))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        py2ipynb(Path(sys.argv[1]))
    else:
        py2rst(Path(sys.argv[1]))
