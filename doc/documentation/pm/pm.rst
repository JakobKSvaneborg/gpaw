.. module:: gpaw.pipekmezey
.. _pipek_mezey_wannier:

=============================
Pipek-Mezey Wannier Functions
=============================

Introduction
============

Pipek-Mezey [#pm]_ Wannier functions (PMWF) is an alternative to the maximally localized 
(Foster-Boys) Wannier functions (MLWF). PMWFs are higly localized orbitals with
chemical intuition where a distinction is maintained between `\sigma` and `\pi` type
orbitals. The PMWFs are as localized as the MLWFs as measured by spread function, 
whereas the MLWFs frequently mix chemically distinct orbitals [#pmwfs]_.


Theoretical Background
======================

In PMWFs the objective function which is maximized is

.. math:: \mathcal{P}(\mathbf{W}) = 
          \sum^{N_\mathrm{occ}}_n \sum_{a}^{N_a}
          \mid Q^a_{nn}(\mathbf{W}) \mid^p

where the quantity `Q^a_{nn}` is the atomic partial charge matrix of atom `a`. `\mathbf{W}`
is a unitary matrix which connects the canonical orbitals `R` to the localized orbitals `n`

.. math:: \psi_n(\mathbf{r}) = \sum_R W_{Rn}\phi_R(\mathbf{r})

The atomic partial charge is defined by partitioning the total electron density, 
in real-space, with suitable atomic centered weight functions

.. math:: n_a(\mathbf{r}) = w_a(\mathbf{r})n(\mathbf{r})

Formulated in this way the atomic charge matrix is defined as

.. math:: Q^a_{mn} = \int \psi^*_m(\mathbf{r})w_a(\mathbf{r})\psi_n(\mathbf r)d^3r
 
where the number of electrons localized on atom `a` follows

.. math:: \sum_n^{N_\mathrm{occ}}Q^a_{nn}=n_a

A choice of Wigner-Seitz or Hirshfeld weight functions is provided, but the
orbital localization is insensitive to the choice of weight function [#genpm]_. 


------------
Localization
------------

The PMWFs is applicable to LCAO, PW and FD mode, and to both open and periodic boundary
conditions. For periodic simulations a uniform Monkhorst-Pack grid must be used.


----------
References
----------

.. [#pm] J. Pipek, P. G. Mezey
         :doi:`A fast intrinsic localization procedure applicable for ab initio and semiempirical linear combination of atomic orbital wave functions <10.1063/1.456588>`,
         *J. Chem. Phys.*, (1989)

.. [#pmwfs] E. Ö. Jónsson, S. Lethola, M. Puska, H. Jónsson
            :doi:`Theory and Application of Generalized Pipek-Mezey Wannier Functions <10.1021/acs.jctc.6b00809>`,
            *J. Chem. Theory Comput.*, (2017)

.. [#genpm] S. Lethola, H. Jónsson
            :doi:`Pipek-Mezey Orbital Localization Using Various Partial Charge Estimates <10.1021/ct401016x>`
            *J. Chem. Theory Comput.*, (2014)
            
