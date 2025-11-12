.. _do_convergence:

==============================================================================
Troubleshooting Excited-State Calculations with Direct Optimization
==============================================================================

Calculations of excited states with :ref:`do` can sometimes present convergence
issues. In this section, we outline a hierarchy of practical strategies to
troubleshoot and improve convergence.

In general, LCAO calculations tend to converge more easily. When convergence
becomes challenging, it is recommended to use :ref:`do-gmf`. If GMF is still
insufficient, the LCAO mode allows for a constrained optimization by freezing
the hole and the excited electron. See the example in :ref:`ppexample`.

FD and PW calculations, on the other hand, may experience convergence problems
for excited states because the unoccupied orbitals are not optimized, unlike in
the LCAO case. In the ground state calculation, the keyword
``converge_unocc`` should remain set to ``False`` (which is the default), since
in the current implementation this option does not behave correctly. It is also
advised to specify a LCAO basis set larger than the default ``dzp`` in the ground
state calculation, even when using FD or PW mode. This will improve the
description of the unoccupied orbitals, leading to better convergence in the
subsequent excited-state calculations.

When specifying an LCAO basis is not sufficient, several scenarios are possible:

* **Using LCAO orbitals:** When the corresponding LCAO calculation is available,
  convergence for FD calculations can be improved by reading in the LCAO
  orbitals::

      H2O, calc_lcao = restart('water_ex_lcao.gpw', txt='-')
      calc_lcao.set_positions(H2O)
      calc_lcao.initialize(H2O)
      calc_lcao.wfs.initialize_wave_functions_from_lcao()

      # -------------------------------------------------
      H2O, calc = restart('water_gs_fd.gpw', txt='Ex_water_fd.txt')
      calc.set_positions(H2O)
      calc.initialize(H2O)

      psit_nG = [calc_lcao.wfs.kpt_u[x].psit_nG.copy()
                 for x in range(len(calc_lcao.wfs.kpt_u))]

      for k, kpt in enumerate(calc.wfs.kpt_u):
          for i in range(len(kpt.psit_nG)):
              kpt.psit_nG[i] = psit_nG[k][i].copy()

      f_sn = excite(calc, 0, 3, spin=(1, 0))

      # -------------------------------------------------
      nbands = 10
      calc.set(
          eigensolver=FDPWETDM(
              excited_state=True,
              need_init_orbs=False))
      prepare_mom_calculation(calc, H2O, f_sn)
      e = H2O.get_potential_energy()

* **Spin-state initialization:** If the triplet state calculation fails to
  converge, but the corresponding mixed-spin excited state has
  converged successfully (or vice versa), it is recommended to read in the
  converged orbitals and then swap the :math:`\alpha` and :math:`\beta`
  orbitals that differ in occupancy between the two spin states. This provides
  a better initial guess.


* **SCF oscillations:** If the SCF steps oscillate around a certain solution but
  do not converge, it is recommended to reduce the step size of
  the inner-loop search (the default value is 0.2)::

      calc.set(
          eigensolver=FDPWETDM(
              max_step_inner_loop=0.1,
              excited_state=True,
              converge_unocc=False,
              need_init_orbs=False))

* **Eigenstate fluctuation:** When the change in the eigenstate reaches less
  than 10^-6 but then starts increasing again, it is advised to:

  first relax the convergence criteria::

      calc.set(
          convergence={
              'energy': 0.0005,      # eV / electron
              'density': 1.0e-4,     # electrons / electron
              'eigenstates': 5.0e-7, # eV**2 / electron
              'bands': 'occupied'},
          eigensolver=FDPWETDM(
              excited_state=True,
              converge_unocc=False,
              need_init_orbs=False))

  and then restart the calculation from the existing ``.gpw`` file, set the
  inner-loop step size to zero (removing the inner loop), and switch the search
  algorithm to steepest descent::

      calc.set(
          convergence={
              'energy': 0.0005,      # eV / electron
              'density': 1.0e-4,     # electrons / electron
              'eigenstates': 8.0e-8, # eV**2 / electron
              'bands': 'occupied'},
          eigensolver=FDPWETDM(
              searchdir_algo='sd',
              max_step_inner_loop=0.0,
              excited_state=True,
              converge_unocc=False,
              need_init_orbs=False))

* **Hybrid-functionals and SIC:** For calculations with hybrid
  functionals and :ref:`sic`, it is advised to first run a calculation with
  a GGA functional, and then use the converged orbitals as the starting guess
  for the hybrid calculation. For example, when the exchange-correlation
  functional is PBE0, first run a PBE calculation and then use the PBE orbitals
  as the starting point::

      H2O, calc = restart('water_gs_pw.gpw', txt='-')
      calc.wfs.initialize_wave_functions_from_restart_file()
      calc.initialize(H2O)

      f_sn = [calc.wfs.kpt_u[x].f_n.copy()
              for x in range(len(calc.wfs.kpt_u))]
      psit_nG = [calc.wfs.kpt_u[x].psit_nG.copy()
                 for x in range(len(calc.wfs.kpt_u))]

      calc2 = GPAW(mode=PW(ecut=1200),
                   xc={'name': 'PBE0', 'backend': 'pw'},
                   basis='aug-cc-pVDZ_PBE.sz',
                   h=0.16,
                   eigensolver=FDPWETDM(converge_unocc=False),
                   mixer={'backend': 'no-mixing'},
                   occupations={'name': 'fixed-uniform'},
                   nbands=9,
                   symmetry='off',
                   spinpol=True,
                   txt='water_M0_pw_pbe0.txt')

      calc2.initialize(H2O)
      calc2.set_positions(H2O)

      for k, kpt in enumerate(calc2.wfs.kpt_u):
          for i in range(len(kpt.psit_nG)):
              kpt.psit_nG[i] = psit_nG[k][i].copy()

      calc2.atoms = H2O
      calc2.calculate(properties=['energy'], system_changes=None)
      gs = H2O.get_potential_energy()







