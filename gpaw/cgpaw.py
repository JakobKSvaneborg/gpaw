from gpaw import no_c_extension


if no_c_extension:
    have_openmp = False

    def get_num_threads():
        return 1
else:
    from _gpaw import *


# def __getattr__(name):
#    return getattr(_gpaw, name)
