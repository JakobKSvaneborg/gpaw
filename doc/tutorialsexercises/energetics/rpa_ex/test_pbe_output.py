"""Test the RPA outputs are consistent and do not change."""

import numpy as np


def test():
    bulk_results, bulk_benchmark = [], [
        5.421,
        6,
        400.0,
        -10.764252623234455,
        13.867018289712249,
    ]
    with open("si.pbe+exx.results.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            x = [float(x) for x in line.strip().split()]
            bulk_results.append(x)

    # make sure results are consistent, the tol here was tested on only one
    # laptop and may to be strict.
    # Loop in case the si.pbe+exx.results.txt file has multiple lines
    for result in bulk_results:
        assert len(result) == 5
        assert np.allclose(result, bulk_benchmark, rtol=1.e-5, atol=1.e-8)


if __name__ == "__main__":
    test()
