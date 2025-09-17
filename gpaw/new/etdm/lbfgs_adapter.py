import numpy as np
from gpaw.new.etdm.searchdir import LBFGS 

class LBFGSAdapter:
    """
    Adapter to allow the new NumPy-based L-BFGS to replace the Vector-based LBFGS.
    """

    def __init__(self, array_shape, kpt_comm, dtype, memory=5):
        self.lbfgs_new = LBFGS(array_shape, kpt_comm, dtype, memory)

    def update(self, psit_unX, pg_unX):
        """
        Convert old-style vectors to NumPy arrays, call new L-BFGS, and convert back.
        """
        # Convert old vectors to NumPy arrays
        a_cur = np.stack([x.data for x in psit_unX])
        g_cur = np.stack([g.data for g in pg_unX])


        # Call the new LBFGS
        p_cur = self.lbfgs_new.update(a_cur, g_cur)

        # Convert back to old-style objects
        p_unX_new = []
        for p_vec, old_vec in zip(p_cur, psit_unX):
            new_vec = old_vec.new()  # create empty vector same type as old
            new_vec.data[:] = p_vec
            p_unX_new.append(new_vec)

        return p_unX_new
