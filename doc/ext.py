try:
    from ase.utils.sphinx import (create_png_files, git_role_tmpl,
                                  mol_role)
except ImportError:
    from ase.utils.sphinx_create_png import (create_png_files, git_role_tmpl,
                                             mol_role)


def git_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    return git_role_tmpl('https://gitlab.com/gpaw/gpaw/blob/master/',
                         role,
                         rawtext, text, lineno, inliner, options, content)


def setup(app):
    app.add_role('mol', mol_role)
    app.add_role('git', git_role)
    create_png_files()
