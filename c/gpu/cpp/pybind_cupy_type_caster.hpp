#pragma once

#include "python_utils.h"
#include "pyarray_utils.hpp"

/* Type caster to convert Cupy.ndarray <-> gpaw::PyDeviceArray (no copy).
* Custom type casters must be added directly to pybind11::detail namespace,
* see https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html */


namespace pybind11
{
namespace detail
{

template<>
struct type_caster<gpaw::PyDeviceArray>
{
    /* Boilerplate macro, creates instance of gpaw::PyDeviceArray called `value`.
    * Note that this calls the default constructor; we must assign correct data to it in the load() function.
    * The name field specifies allowed types to convert from/to.
    * There are fancier ways of specifying the name in newer pybind11, see docs, but this method works with older version too. */
    PYBIND11_TYPE_CASTER(gpaw::PyDeviceArray, _("cupy.ndarray"));

    /* C++ -> Python conversion. FIXME: can't easily have this as Cupy provides no C++ interface.
    * So either we call Cupy's Python API from here, or simply don't allow conversion like this.
    * For not, don't allow it. */
    static handle cast(const gpaw::PyDeviceArray& array, return_value_policy policy, handle parent)
    {
        throw std::invalid_argument("Conversion not allowed: gpaw::PyDeviceArray to Cupy ndarray.");
    }

    /* Python -> C++ conversion. Must return false on failure, which raises a TypeError on Python side.
    */
    bool load(handle src, bool implicit_convert)
    {
        if (!gpaw::is_cupy_array(src))
        {
            return false;
        }

        value.from_cupy(src);
        return true;
    }
};

} // namespace detail
} // namespace pybind11
