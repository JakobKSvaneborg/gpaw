.. _ipad:

==============
 Using an iPad
==============

Using an iPad for the computer exercises is **not recommended,** mainly because you cannot run the ASE graphical user interface.  But it is a possibility if there are no other options.

Terminal app
============

A terminal app that is known to work is Termius.

First you need to install your secret key (guest accounts only, DTU
users need to generate a new key).  Open a browser, go to https://www.student.dtu.dk/~dccidmgr/CAMD24/ then add your user name to
the end of the URL and press enter.  You will now have to enter your
user name and password.  Click on the provided link (saving it will
not help, you need to use cut-and-paste).  Once the key is displayed
(a lot of randomish letters between a header and a footer line) select
it all including header and footer, and copy it.

Then start Termius, in the left menu click *Keychain*.  Choose the
plus sign at the top of the page, and choose *Paste key*.  Paste the
key you copied from the browser into the field called *Private Key*.
Give your new key a label (e.g. DTU), and if you feel paranoid you can
also set a password for it.  Save it.

Create a login by choosing ``Hosts`` in the sidebar, then select the Plus sign at the top.  Select "New Host"

* Label: DTU gbar (or whatever you want)
* IP or Hostname: login.gbar.dtu.dk
* Use SSH should be turned on.
* Username: Your DTU user name
* Password: Your DTU password
* Key: Select the private key you just created.

Click on the new profile, you should be logged in.

Now follow the instructions for :ref:`logging in the first time with a Mac <setuplinmac>`, with the important change that the command ``linuxsh -X`` should be replaced with ``linuxsh``.  The omitted ``-X`` makes it possible for the graphical user interface to show atomic structures on your screen, that does not work on an iPad and a workaround is described below.

Connecting to Jupyter Notebooks
===============================

Then follow the guide for :ref:`Starting and accessing a Jupyter Notebook <accesslinmac>`  When you start the Jupyter server with the ``camdnotebook`` command, you will be warned that X11 forwarding is not set up.  Ignore the warning and continue by pressing ENTER.

You now need to set up port forwarding.  Make a note of the computer name and host number, i.e. n-62-27-19 and 40042.  Then click on the button with a < sign in the upper right corner.

* Click on Port Forwarding.  Then click on the Plus or the New Rule button
* Select type "local" and click on Continue 
* Set local port to the port number you noted (e.g. 40042, it will be
  different for you).  Leave the other field blank.  Press Continue
* Select a host.  Choose the same setup that you made when you logged in.
* Destination address is the host name you noted, i.e. n-62-27-19 (it will be different for you!)
* Destination port is the port number you noted, i.e. 40042 (it will also be different for you)
* Click Done, then Save

Then click on the new forwarding button to start it.

**You cannot edit the forwarding rule.  When you log out and in again and get a new host and portnumber, you have to define a new rule.  Make sure the old one is not running (maybe even delete it).**

Now click on your terminal connection to go back to the terminal window.

Using the notebook
==================

You need to have Termius and your browser running simultaneously.
Click on the three dots on the top of the window, and choose the split
view.  Then open your browser.  You should now have Termius and the
browser running side by side.  In the terminal, select the line
starting with ``https://127.0.0.1`` and paste it into your browser (or
open it, if Termius offers it).  

Viewing atomic structures
=========================

The ``view(atoms)`` command in ASE will not work for you, as you do not have an  X11 server on your ipad.  Instead use the ``plot_atoms`` command::

  from ase.visualize.plot import plot_atoms
  plot_atoms(atoms)

You rotate the atoms like this (45 degrees along the x-axis)::

  plot_atoms(atoms, rotation='45x')


  
  

  
