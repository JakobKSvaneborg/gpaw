import cupy
from . import magma as magma

def test_array_metadata(array: object, same_array: cupy.ndarray) -> None:
    """test_array_metadata(array: object, same_array: cupy.ndarray) -> None

    Tests that the array type caster correctly copies metadata from cupy ndarray. Use by passing the same array as both inputs. Raises RuntimeError on failure.
    """
def test_cupy_input(a: cupy.ndarray) -> None:
    """test_cupy_input(a: cupy.ndarray) -> None

    Simply tests that our type caster works, ie. that we can pass a Cupy ndarray to a C++ function accepting gpaw::PyDeviceArray
    """
