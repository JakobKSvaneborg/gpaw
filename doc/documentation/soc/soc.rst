.. _soc:

Spin-orbit coupling and non-collinear calculations
==================================================

.. _spin direction constrained dft:

Spin-direction constrained DFT
------------------------------

Suppose we want to constrain the direction of the spin magnetic moment
`\vec{m}_a` at some atomic site `a` along some target direction defined by
the unit vector `\vec{\hat{u}}_a`. This can be done by
introducing the penalty functional

.. math::

    E_\mathrm{cDFT}=\Lambda
    \left({\vphantom{\frac{1}{1}}\vec{m}_a-\vec{\hat{u}}_a\left({\vec{\hat{u}}_a\cdot\vec{m}_a}\right)}\right)^2
    =\Lambda\left({\vphantom{\frac{1}{1}}\vec{m}_a\cdot\vec{m}_a-\left({\vec{\hat{u}}_a\cdot\vec{m}_a}\right)^2}\right),

with the penalty `\Lambda` in units of eV pr.  `\mu_\mathrm{B}^2`.
We take advantage of the PAW formalism to
define the local magnetic moment at site `a` which we want to constrain.
Reminder, the atomic spin density matrices are defined through

.. math::

    D^a_{x,i_1i_2}&=\sum_{\vec{k}n}f_{\vec{k}n}\left({\braket{\widetilde{\psi}_{\vec{k}\uparrow n}}{\widetilde{p}^a_{i_1}}\braket{\widetilde{p}^a_{i_2}}{\widetilde{\psi}_{\vec{k}\downarrow n}}+\braket{\widetilde{\psi}_{\vec{k}\downarrow n}}{\widetilde{p}^a_{i_1}}\braket{\widetilde{p}^a_{i_2}}{\widetilde{\psi}_{\vec{k}\uparrow n}}}\right), \\
    D^a_{y,i_1i_2}&=-i\sum_{\vec{k}n}f_{\vec{k}n}\left({\braket{\widetilde{\psi}_{\vec{k}\uparrow n}}{\widetilde{p}^a_{i_1}}\braket{\widetilde{p}^a_{i_2}}{\widetilde{\psi}_{\vec{k}\downarrow n}}-\braket{\widetilde{\psi}_{\vec{k}\downarrow n}}{\widetilde{p}^a_{i_1}}\braket{\widetilde{p}^a_{i_2}}{\widetilde{\psi}_{\vec{k}\uparrow n}}}\right), \\
    D^a_{z,i_1i_2}&=\sum_{\vec{k}n}f_{\vec{k}n}\left({\braket{\widetilde{\psi}_{\vec{k}\uparrow n}}{\widetilde{p}^a_{i_1}}\braket{\widetilde{p}^a_{i_2}}{\widetilde{\psi}_{\vec{k}\uparrow n}}-\braket{\widetilde{\psi}_{\vec{k}\downarrow n}}{\widetilde{p}^a_{i_1}}\braket{\widetilde{p}^a_{i_2}}{\widetilde{\psi}_{\vec{k}\downarrow n}}}\right).

If we only wish to constrain the PAW part of the magnetic moment at site `a`, we can write (neglecting the negative sign as is standard in GPAW)

.. math::

    m_{a,x}=\mu_\mathrm{B}\sum_{i_1i_2}D^a_{x,i_1i_2}N_{i_1i_2},\quad
    m_{a,y}=\mu_\mathrm{B}\sum_{i_1i_2}D^a_{y,i_1i_2}N_{i_1i_2},\quad
    m_{a,z}=\mu_\mathrm{B}\sum_{i_1i_2}D^a_{z,i_1i_2}N_{i_1i_2},

with

.. math::

    N_{i_1i_2}=
    \braket{\phi^a_{i_1}}{\phi^a_{i_2}}_{\hspace{-2pt}\mathrm{PAW}}=
    \delta_{l_1l_2}\delta_{m_1m_2}\braket{R^a_{j_1}}{R^a_{j_2}}_{\hspace{-2pt}\mathrm{PAW}}

i.e. the inner product of partial waves `\phi^a_i` is restricted within PAW spheres.\par\bigskip
Expanding the dot products in the penalty functional, we get

.. math::

    E_\mathrm{cDFT}=\Lambda&\left(\vphantom{\frac{1}{1}}
    \left({1-\hat{u}_{a,x}^2}\right)m_{a,x}^2 + \left({1-\hat{u}_{a,y}^2}\right)m_{a,y}^2 + \left({1-\hat{u}_{a,z}^2}\right)m_{a,z}^2
    \right. \\
    &\left.\vphantom{\frac{1}{1}}-2\left({\hat{u}_{a,x}\hat{u}_{a,y} m_{a,x}m_{a,y}
    +\hat{u}_{a,x}\hat{u}_{a,z} m_{a,x}m_{a,z}
    +\hat{u}_{a,y}\hat{u}_{a,z} m_{a,y}m_{a,z}}\right) \right),

and since we have

.. math::

    \frac{\partial m_{a,x}}{\partial D^a_{x,i_1i_2}}=\mu_\mathrm{B}N_{i_1i_2},\qquad
    \frac{\partial m_{a,y}}{\partial D^a_{y,i_1i_2}}=\mu_\mathrm{B}N_{i_1i_2},\qquad
    \frac{\partial m_{a,z}}{\partial D^a_{z,i_1i_2}}=\mu_\mathrm{B}N_{i_1i_2},

we can use the chain rule to get the additions to the atomic Hamiltonians

.. math::

    H^a_{x,i_1i_2}=\frac{\partial E_\mathrm{cDFT}}{\partial D^a_{x,i_1i_2}}=2\Lambda\mu_\mathrm{B}&
    \left[{\vphantom{\frac{1}{1}}
    \left({1-\hat{u}_{a,x}^2}\right)m_{a,x}-\hat{u}_{a,x}\left({\hat{u}_{a,y}m_{a,y}+\hat{u}_{a,z}m_{a,z}}\right)}\right]N_{i_1i_2}, \\
    H^a_{y,i_1i_2}=\frac{\partial E_\mathrm{cDFT}}{\partial D^a_{y,i_1i_2}}=2\Lambda\mu_\mathrm{B}&\left[{\vphantom{\frac{1}{1}}
    \left({1-\hat{u}_{a,y}^2}\right)m_{a,y}-\hat{u}_{a,y}\left({\hat{u}_{a,x}m_{a,x}+\hat{u}_{a,z}m_{a,z}}\right)}\right]N_{i_1i_2}, \\
    H^a_{z,i_1i_2}=\frac{\partial E_\mathrm{cDFT}}{\partial D^a_{z,i_1i_2}}=2\Lambda\mu_\mathrm{B}&\left[{\vphantom{\frac{1}{1}}
    \left({1-\hat{u}_{a,z}^2}\right)m_{a,z}-\hat{u}_{a,z}\left({\hat{u}_{a,x}m_{a,x}+\hat{u}_{a,y}m_{a,y}}\right)}\right]N_{i_1i_2},

.. autoclass:: gpaw.new.constraints.SpinDirectionConstraint

See :git:`gpaw/test/noncollinear/test_spin_dir_constraint.py`
for how to use the
:class:`~gpaw.new.constraints.SpinDirectionConstraint` extension.


2D example
----------

https://journals.aps.org/prb/abstract/10.1103/PhysRevB.62.11556

.. literalinclude:: VCl2.py
.. literalinclude:: plot.py

.. figure:: mag2d.png
.. figure:: mag1d.png

Experiential:

    https://doi.org/10.3390/cryst7050121

Theoretical:

    https://doi.org/10.1088/0953-8984/10/22/004

DFT:

    https://doi.org/10.1063/1.4791437
