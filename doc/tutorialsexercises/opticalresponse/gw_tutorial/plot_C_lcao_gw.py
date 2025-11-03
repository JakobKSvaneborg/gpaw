import pickle
from matplotlib import pyplot as plt
import numpy as np

pw_file = "C-g0w0-pw_results_GW.pckl"
lcao_file = "C-g0w0-lcao_results_GW.pckl"

omega = np.linspace(-50, 75, 500)

with open(pw_file, "rb") as f:
    pw = pickle.load(f)
with open(lcao_file, "rb") as f:
    lcao = pickle.load(f)

sigma_pw = pw["sigma_eskwn"][0, 0, 0, :, :]
sigma_lcao = lcao["sigma_eskwn"][0, 0, 0, :, :]

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(omega, np.imag(sigma_pw[:, 0]), linewidth=2, label="VB PW")
ax.plot(omega, np.imag(sigma_pw[:, 1]), linewidth=2, label="CB PW")
ax.plot(omega, np.imag(sigma_lcao[:, 0]), linewidth=2, label="VB LCAO")
ax.plot(omega, np.imag(sigma_lcao[:, 1]), linewidth=2, label="CB LCAO")
ax.set_title("Diamond")
ax.set_xlabel(r"$\omega\,[\mathrm{eV}]$")
ax.set_ylabel(r"${\rm Im}\,\Sigma(\omega)\,[\mathrm{eV}]$")
ax.legend(frameon=False)
fig.savefig("C_Im.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(omega, np.real(sigma_pw[:, 0]), linewidth=2, label="VB PW")
ax.plot(omega, np.real(sigma_pw[:, 1]), linewidth=2, label="CB PW")
ax.plot(omega, np.real(sigma_lcao[:, 0]), linewidth=2, label="VB LCAO")
ax.plot(omega, np.real(sigma_lcao[:, 1]), linewidth=2, label="CB LCAO")
ax.set_title("Diamond")
ax.set_xlabel(r"$\omega\,[\mathrm{eV}]$")
ax.set_ylabel(r"${\rm Re}\,\Sigma(\omega)\,[\mathrm{eV}]$")
ax.legend(frameon=False)
fig.savefig("C_Re.png", dpi=300, bbox_inches="tight")
plt.close(fig)
