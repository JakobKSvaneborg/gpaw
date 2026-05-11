import re
from collections.abc import Generator
from pathlib import Path

import numpy as np


def find_created_files(root: Path = Path()) -> Generator[Path, None, None]:
    for path in root.glob('**/*.py'):
        if path.parts[0] == 'build':
            continue
        filenames = []
        for line in path.read_text().splitlines():
            if not line.startswith('# web-page:'):
                break
            filenames += line.split(':')[1].split(',')
        for name in filenames:
            name = name.strip()
            yield path.with_name(name)


def collect_files_for_web_page(fro: Path, to: Path) -> None:
    for path in find_created_files(fro):
        p = to / path.relative_to(fro)
        print(path, p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(path.read_bytes())


def compare_all_files(folder1: Path,
                      folder2: Path) -> None:
    errors = []
    missing = []
    ok = 0
    for p1 in find_created_files(folder1):
        p2 = folder2 / p1.relative_to(folder1)
        if not p1.is_file():
            missing.append(p1)
            continue
        if not p2.is_file():
            missing.append(p2)
            continue
        err = compare_files(p1, p2)
        if err:
            errors.append((p1, p2))
        else:
            ok += 1
    print(ok, len(errors), len(missing))
    for p1, p2 in errors:
        if p1.suffix in {'.png', '.svg'}:
            pass  # rint(f'eog {folder1 / path} {folder2 / path}')
        else:
            print(f'meld {p1} {p2}')


def compare_files(p1: Path, p2: Path) -> float:
    if p1.suffix in {'.png', '.svg'}:
        return compare_images(p1, p2)
    if p1.suffix == '.db':
        return 0.0
    return compare_text(p1, p2)


def compare_images(p1: Path, p2: Path) -> float:
    import PIL.Image as pil
    a1, a2 = (np.asarray(pil.open(p)) for p in [p1, p2])
    if a1.shape != a2.shape:
        print('Wrong size', a1.shape, a2.shape)
        return 10.0
    d = a1 - a2
    N = 4
    n1, n2, _ = d.shape
    d = d[:n1 // N * N, :n2 // N * N]
    d = d.reshape((n1 // N, N, n2 // N, N, -1))
    d = d.mean(axis=(1, 3))
    error = abs(d).mean() / 255 / 4
    if error > 0.00012:
        return error
    return 0.0


def compare_text(p1: Path, p2: Path) -> float:
    t1 = p1.read_text()
    t2 = p2.read_text()
    if t1 == t2:
        return 0.0

    # (?s:...) means match also newlines
    for r in [r'(?s:User: .*OMP_NUM_THREADS)',
              r'(?s:Timing:.*)',
              r'Lattice=".*pbc="',
              r'Process memory now:.*',
              r'iter: .*',
              r'20..-..-.. .*:..:..\.\d*',  # date
              r'Calculating spectrum .*',
              r'Spectrum calculated .*']:
        t1 = re.sub(r, '', t1)
        t2 = re.sub(r, '', t2)
    if t1 == t2:
        return 0.0
    lines1 = t1.splitlines()
    lines2 = t2.splitlines()
    sep = ',' if p1.suffix == '.csv' else None

    rtol = 0.002
    atol = 1e-5
    if 'lcao-time' in p1.name:
        atol = 1.5
    elif p1.name == 'TS.xyz':
        atol = 0.02

    for l1, l2 in zip(lines1, lines2):
        if l1 == l2:
            continue
        words1 = l1.split(sep)
        words2 = l2.split(sep)
        if len(words1) != len(words2):
            print(l1, l2, words1, words2, sep)
            return 1.0
        for w1, w2 in zip(words1, words2):
            try:
                f2 = float(w2)
            except ValueError:
                if w1 != w2:
                    print(l1, l2, w1, w2)
                    return 1.0
            else:
                f1 = float(w1)
                if not np.allclose(f1, f2, atol=atol, rtol=rtol):
                    error = f1 - f2
                    print(l1, l2, f1, f2, error)
                    return 1.0
    return 0.0


if __name__ == '__main__':
    import sys
    compare_all_files(Path(sys.argv[1]), Path(sys.argv[2]))
    # collect_files_for_web_page(Path(sys.argv[1]), Path(sys.argv[2]))
