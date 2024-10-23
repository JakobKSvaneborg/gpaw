import subprocess
from pathlib import Path


def setup(app):
    # Get png and csv files and other stuff from the AGTS scripts that run
    # every weekend:
    data = Path('/tmp/gpaw-web-page-data')
    if data.is_dir():
        subprocess.run(f'cd {data} && git pull', shell=True)
    else:
        repo = 'https://gitlab.com/gpaw/gpaw-web-page-data.git/'
        subprocess.run(f'cd /tmp && git clone {repo}', shell=True)

    doc = Path()
    for path in (data / 'doc/').glob('**/*.*'):
        fro = doc / path.relative_to(data / 'doc/')
        if not fro.is_file():
            print(fro, '->', path)
            fro.symlink_to(path)


if __name__ == '__main__':
    setup(None)
