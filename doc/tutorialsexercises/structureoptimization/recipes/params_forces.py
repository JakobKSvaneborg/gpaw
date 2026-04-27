# fast forces
calculator_params = {
    "xc": "PBE",
    "basis": "dzp",
    "mode": {"name": "lcao"},
    "kpts": {"size": [1, 1, 1],
             "gamma": True},
    "convergence": {"density": 1e-3,
                    "forces": 1e-2},
    "occupations": {"name": "fermi-dirac",
                    "width": 0.05},
    "mixer": {"method": "fullspin",
              "backend": "pulay"},
    "txt": "rlx.txt",
}
