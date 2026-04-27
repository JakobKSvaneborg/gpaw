#pragma once

#include "gpaw_utils.h"
#include "extensions.h"

/* PATCH: elpa_strerr() is missing extern "C" in their headers. Seems to be
fixed in early 2025 but we fix it manually here for backwards compatibility.
See commit 4e0d286eb91de0cd5945798c387232dc81e2b7ed in ELPA repo.

FIXME: should not be GPAW's responsibility to patch this!
*/
CLINKAGE const char *elpa_strerr(int elpa_error);

#include <elpa/elpa.h>
