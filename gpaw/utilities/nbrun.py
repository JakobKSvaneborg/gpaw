"""Code for summerschool notebooks."""
import sys
from pathlib import Path


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
                else:
                    output.append('    ' + line)

        output.append('')

    outpath = path.with_suffix('.rst')
    assert not outpath.is_file(), outpath
    outpath.write_text(
        '\n'.join(output))


if __name__ == '__main__':
    py2rst(Path(sys.argv[1]))
