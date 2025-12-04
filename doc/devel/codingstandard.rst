.. _codingstandard:

==================
Coding Conventions
==================

Python Coding Conventions
=========================

The rules are almost identical
to those used by the `Docutils project`_:

.. attention::

  Contributed code will not be refused merely because it does not
  strictly adhere to these conditions; as long as it's internally
  consistent, clean, and correct, it probably will be accepted.  But
  don't be surprised if the "offending" code gets fiddled over time to
  conform to these conventions.

The project follows the generic coding conventions as
specified in the `Style Guide for Python Code`_ and `Docstring
Conventions`_ PEPs, clarified and extended as follows:

* Use ``'single quotes'`` for string literals, and
  ``"""triple double quotes"""`` for :term:`docstring`\ s.
  Double quotes are OK for something like ``"don't"``.

* No trailing commas.

* No hanging end-parenthesis.

* Do not use "``*``" imports such as ``from module import *``.  Instead,
  list imports explicitly.

* Use 4 spaces per indentation level.  No tabs.

* Read the *Whitespace in Expressions and Statements*
  section of PEP8_.

* Avoid `trailing whitespaces`_.

* No one-liner compound statements (i.e., no ``if x: return``: use two
  lines).

* Maximum line length is 78 characters.

* Use "StudlyCaps" for class names.

* Use "lowercase" or "lowercase_with_underscores" for function,
  method, and variable names.  For short names,
  joined lowercase may be used (e.g. "tagname").  Choose what is most
  readable.

* No single-character variable names, except indices in loops
  that encompass a very small number of lines
  (``for i in range(5): ...``).

* Avoid lambda expressions.  Use named functions instead.

* Avoid functional constructs (filter, map, etc.).  Use list
  comprehensions instead.


.. _Style Guide for Python Code:
.. _PEP8: https://www.python.org/dev/peps/pep-0008/
.. _Docstring Conventions: https://www.python.org/dev/peps/pep-0257/
.. _Docutils project: http://docutils.sourceforge.net/docs/dev/policies.html
                      #python-coding-conventions
.. _trailing whitespaces: http://www.gnu.org/software/emacs/manual/html_node/
                          emacs/Useless-Whitespace.html

Example:

.. literalinclude:: coding_style.py


General advice
--------------

 * Get rid of as many ``break`` and ``continue`` statements as possible.

 * Write short functions.  All functions should fit within a standard screen.

 * Use descriptive variable names.


Writing documentation in the code
---------------------------------

ASE follows the NumPy/SciPy convention for docstrings:

  https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard


C-code
======

Code C in the C99 style::

  for (int i = 0; i < 3; i++) {
      double f = 0.5;
      a[i] = 0.0;
      b[i + 1] = f * i;
  }

and try to follow PEP7_.

Use **M-x c++-mode** in emacs.

.. _PEP7: https://www.python.org/dev/peps/pep-0007
