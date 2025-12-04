from pathlib import Path


def setup(app):
    # Get png and csv files and other stuff from the AGTS scripts that run
    # every weekend:
    import gpaw_web_page_data
    data = Path(gpaw_web_page_data.__file__).parent
    print('Using gpaw-web-page-data from', data)
    doc = Path()
    for path in data.glob('**/*.*'):
        if path.name.startswith('_'):
            continue
        fro = doc / path.relative_to(data)
        if not fro.is_file():
            print(fro, '->', path)
            if fro.is_symlink():
                fro.unlink()
            fro.symlink_to(path)


if __name__ == '__main__':
    setup(None)
