.. _development workflow:

====================
Development workflow
====================

.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _venv: https://docs.python.org/3/library/venv.html#module-venv
.. _pip: https://pip.pypa.io/
.. _git: https://git-scm.com/
.. _GitLab issues: https://gitlab.com/gpaw/gpaw/issues
.. _pytest: https://docs.pytest.org/en/6.2.x/

.. contents::

.. seealso::

   * :ref:`writing documentation`
   * :ref:`testing`


Setting up your development environment
=======================================

Make a `virtual environment <venv>`_::

 $ mkdir devel
 $ cd devel
 $ unset PYTHONPATH
 $ python3 -m venv venv
 $ source venv/bin/activate  # venv/bin/ is now first in $PATH
 $ pip install --upgrade pip

Install master branch of ASE_ in *editable* mode::

 $ git clone git@gitlab.com:ase/ase
 $ pip install --editable ase/

Same thing for GPAW::

 $ git clone git@gitlab.com:gpaw/gpaw
 $ echo "noblas = True; nolibxc = True" > gpaw/siteconfig.py
 $ pip install -e gpaw

.. note::

    Here we used a simple ``siteconfig.py`` that *should* always work:

    * ``noblas = True``: Use the BLAS library built into  NumPy_
      (usually OpenBLAS).
    * ``nolibxc = True``: Use GPAW's own XC-functionals
      (only LDA, PBE, revPBE, RPBE and PW91).

    See :ref:`siteconfig` for details.


Run the tests
=============

The test-suite can be found in :git:`gpaw/test/`.  Run it like this::

 $ pip install pytest-xdist
 $ cd gpaw
 $ pytest -n4

And with MPI (2, 4 and 8 cores)::

 $ mpiexec -n 2 pytest

.. warning::

   This will take forever!  It's a good idea to learn and master pytest_'s
   command-line options for selecting the subset of all the tests that are
   relevant.


Creating a merge request
========================

.. _become a member:

.. important::

   Request to become a member of the ``gpaw`` project on GitLab
   `here <https://gitlab.com/gpaw/gpaw/>`__.  This will
   allow you to push branches to the central repository (see below).

Create a branch for your changes::

 $ cd gpaw
 $ git switch -c fix-something

.. note::

   ``git switch -c fix-something`` is the same as any of these:

   * ``git branch fix-something && git switch fix-something``
   * ``git branch fix-something && git checkout fix-something``
   * ``git checkout -b fix-something``

   :xkcd:`More git-tricks <1597>`.

Make some changes and commit::

 $ git add file1 file2 ...
 $ git commit -m "Short summary of changes"

Push your branch to GitLab::

 $ git push --set-upstream origin fix-something

and click the link to create a merge-request (MR).  Mark the MR as DRAFT to
signal that it is work-in-progress and remove the DRAFT-marker once the MR
is ready for code review.

Every time you push your local repository changes upstream to the remote
repository, you will trigger a continuous integration (CI) runner on the
GitLab servers.  The script that runs in CI is :git:`.gitlab-ci.yml`.
Here is a very short summary of what happens in CI:

* install the code
* ``pytest -m ci``: small selection of fast tests
* ``mypy -p gpaw``: `Static code analysis`_ (type hints)
* ``flake8``: pyflakes + pycodestyle (pep8) = flake8_

If CI fails, you will have to fix things and push your changes.

It's a good idea to also run the CI-checks locally::

 $ cd gpaw
 $ pip install -e .[devel]
 $ flake8 ...
 $ mypy ...
 $ pytest ...
 $ # fix things
 $ git add ...
 $ git commit ...
 $ git push  # Git now knows your upstream

.. tip::

   You can use ``git push -i ci.skip`` if you want to skip CI.


.. _Static code analysis: https://mypy.readthedocs.io/en/stable/
.. _flake8: https://flake8.pycqa.org/en/latest/


.. _a good mr:

How to write a good MR
======================

A good MR

* is short
* does one thing

For MRs with code changes:

* make sure there is a test that covers the new/fixed code
* make sure all variable and functions have descriptive names.
* remember docstrings - if needed
  (no need for an ``add_numbers()`` function to have an
  ``"""Add numbers."""`` docstring).

For MRs with documentation changes,
build the HTML-pages and make sure everything looks OK::

 $ cd gpaw
 $ pip install -e .[docs]
 $ cd doc
 $ make
 $ make browse


.. _get your mr merged:

How to get your MR merged
=========================

* Is your MR branch in your own fork?  Close the MR, push your branch to
  the main repository and open a new MR from there.
  This will allow our CI-runner to test your MR.  You will need to be a
  member of the gpaw project in order to push branches to the main repository
  (see :ref:`here <become a member>`).

* Is it still marked as a draft?
  If so, make sure it is finalized and remove the draft indicator.
  Or if you want feedback before you feel the MR is finished, please
  ask explicitly for review and tag one of the maintainers.

* Is the pipeline passing, including all flake and typing tests?
  If not, make sure that pipeline is passing.

* Does the MR have an accurate title and a description including motivation
  for the change?
  If it is a bug fix, or just few lines, less is required.  However, if it
  is a full feature, the reviewer should be able to get a good overview.

* Have you selected a reviewer?
  If not, please select one from the following list:

  * Jens Jørgen Mortensen (``@jensj``)
  * Ask Hjorth Larsen (``@askhl``)
  * Mikael Kuisma (``@mikaelkuisma``)
  * Tuomas Rossi (``@trossi``)

* Make sure you don't have the ball.
  Perhaps there are comments by the reviewer in the merge request you have not
  answered to.

* Is your merge request more than 50 commits behind from master?
  If so, merge master, and run the full test suite (including gpw-files and
  nightly-mpi-* tests)?

* Does the reviewer have the ball?
  We are sometimes busy, and also human, and we might just not simply see
  the review request, or maybe we just procrastinate.  If you have
  answered all the comments, or are waiting for the first review, and it
  has been more than a week: Please send a friendly reminder by tagging in
  git.  Has it been more than two weeks?  Please send an e-mail to the
  reviewer and ask about the situation.  Be active.

* Need help with git or gitlab: Just ask!

Some developments of big projects go on for over a year, and it might get
increasingly difficult to keep merging master with merge conflicts, or
even worse, the branch could diverge from master.  It is ok to merge
incomplete features, provided that it is obvious to the user, that they
are not ready for production yet and the code has appropriate warnings
and assertions. That
way, you can still add tests, make sure your development keeps track with
the developments of the other parts of the code.

If you decide to have your code in a separate package, but you would be
relying on some part of GPAW's functionality, you can
create a merge requests which tests, that GPAW works and will
continue to work the way your interface needs in the future.  We may
choose to change it anyway, but at least then you would be notified about
the incompatibility, and you can modify your side of the package bundle.
