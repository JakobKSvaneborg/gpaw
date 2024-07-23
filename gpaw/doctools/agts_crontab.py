import os
from pathlib import Path

import numpy as np


def send_email(tasks):
    import smtplib
    from email.message import EmailMessage

    txt = 'Hi!\n\n'
    for task in tasks:
        if task.state in {'FAILED', 'CANCELED', 'TIMEOUT', 'MEMORY'}:
            id, dir, name, res, age, status, t, err = task.words()
            txt += ('test: {}/{}@{}: {}\ntime: {}\nerror: {}\n\n'
                    .format(dir.split('agts/gpaw')[1],
                            name,
                            res[:-1],
                            status,
                            t,
                            err))
    txt += 'Best regards,\nNiflheim\n'

    msg = EmailMessage()
    msg.set_content(txt)
    msg['Subject'] = 'Failing Niflheim-tests!'
    msg['From'] = 'agts@niflheim.dtu.dk'
    msg['To'] = 'jjmo@dtu.dk'
    s = smtplib.SMTP('smtp.ait.dtu.dk')
    s.send_message(msg)
    s.quit()


def find_created_files(root: Path = Path()):
    names = set()
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
            if name in names:
                raise RuntimeError(
                    f'The name {name!r} is used in more than one place!')
            names.add(name)
            yield path.with_name(name)


def collect_files_for_web_page():
    os.chdir('gpaw/doc')
    folder = Path('agts-files')
    if not folder.is_dir():
        folder.mkdir()
    for path in find_created_files():
        print(path)
        (folder / path.name).write_bytes(path.read_bytes())


def compare_all_files(references: Path,
                      root: Path):
    for path in find_created_files(root):
        if not path.is_file():
            continue
        ok = compare_files(path, references / path.name)
        if not ok:
            print(path, references / path.name)


def compare_files(p1, p2):
    if p1.suffix == '.png':
        return compare_images(p1, p2)
    if p1.suffix == '.db':
        return True
    return p1.read_text() == p2.read_text()


def compare_images(p1, p2):
    import PIL.Image as pil
    a1, a2 = (np.asarray(pil.open(p)) for p in [p1, p2])
    if a1.shape != a2.shape:
        print(a1.shape, a2.shape)
        return False
    error = abs(a1 - a2).mean() / 255 / 3
    if error > 1e-3:
        print(error)
        return False
    return True


if __name__ == '__main__':
    import sys
    compare_all_files(Path(sys.argv[1]), Path(sys.argv[2]))
