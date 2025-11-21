/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "bmgs.h"
#include "../gpaw_utils.h"

#if GPAW_CPP
#include <algorithm>
#endif

#include <string.h>

void Z(bmgs_zero)(TGPAW* a, const int n[3], const int c[3],
		  const int s[3])
{
  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < s[0]; i0++)
    {
      for (int i1 = 0; i1 < s[1]; i1++)
	{

  #if GPAW_CPP && defined(BMGSCOMPLEX)
	  // Patch for complex TGPAW in C++ mode. Now TGPAW = std::complex<double>, and
    // the compiler warns about using memset on non-POD objects.
    // For std::complex memset should still be fine (same data layout as C-style complex number),
    // but we do this with std::fill instead to avoid a compiler warning.
    std::fill(a, a + s[2], TGPAW{0., 0.});
  #else
    memset(a, 0, s[2] * sizeof(TGPAW));
  #endif

	  a += n[2];
	}
      a += n[2] * (n[1] - s[1]);
    }
}
