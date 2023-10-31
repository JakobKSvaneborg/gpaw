.. _sic:

======================================================
The Perdew-Zunger Self-Interaction Correction (PZ-SIC)
======================================================
The self-interaction corrected density functional
with PZ-SIC [#Perdew]_  has the following form:

.. math:: E^{PZ-SIC}[\{n_i\}] = E^{DFA}[n] - \beta \sum_{i=1}^{N} \left(E_H[n_i] + E^{DFA}_{xc}[n_i]\right)

here `E^{DFA}` - density functional approximation,
`E_H` - Hartree energy and 
`E^{DFA}_{xc}` - exchange correlation part of density functional approximation. 
`n` is the total density and `n_i = |\psi_i (\mathbf{r})|^{2}` - orbital density. 
`\beta` is a scaling factor.

The SIC functional is not a unitary invariant functional and is dependent on orbital densities. 
Therefore, the :ref:`fully variational approach <directmin>` [#Ivanov2021pwfd]_, [#Ivanov2021]_ is used to find the optimal
orbitals which provide the ground state energy.

Example
--------
The implementation support all three modes (PW, FD, and LCAO) using
with direct minimization described :ref:`here <directmin>` .
Since the functional is not a unitary invariant functional,
it is necessary to employ complex orbitals to find the lowest energy state.
Here is an example using FD mode:

.. literalinclude:: sic_example_fd.py

To use PW mode, just import PW mode and replace FD with PW.
While here is the example for LCAO mode:

.. literalinclude:: sic_example_lcao.py

If you use this module, please refer to implementation papers Refs. [#Ivanov2021pwfd]_, [#Ivanov2021]_.

References
----------

.. [#Perdew] J. P. Perdew and Alex Zunger
             *Phys. Rev. B* **23**, 5048 (1981)

.. [#Ivanov2021pwfd] A. V. Ivanov, G. Levi, E.Ö. Jónsson, and H. Jónsson,
           *J. Chem. Theory Comput.*, **17**, 5034, (2021).

.. [#Ivanov2021] A. V. Ivanov, E.Ö. Jónsson, T. Vegge, and H. Jónsson,
         *Comput. Phys. Commun.*, **267**, 108047 (2021).
