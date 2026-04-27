# accurate stress
calculator_params = {
    "xc": "PBE",
    "basis": "dzp",
    "mode": {"name": "pw",
             "ecut": 600},
    "kpts": {"size": [5, 5, 5],
             "gamma": True},
    "convergence": {"density": 1e-6,
                    "forces": 1e-4},
    "occupations": {"name": "fermi-dirac",
                    "width": 0.05},
    "mixer": {"method": "fullspin",
              "backend": "pulay"},
    "txt": "rlx.txt",
}
