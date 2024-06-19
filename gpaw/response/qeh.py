import numpy as np
from ase.units import pi, Bohr
from qeh.bb_calculator.chicalc import ChiCalc, QPoint
from gpaw.response.df import DielectricFunction

class GPAW_ChiCalc(ChiCalc):
    def __init__(self,
                 df : DielectricFunction,
                 qinf_min : float = 1e-4,  
                 qinf_max : float = 0.5,
                 nq_inf : int = 10,
                 direction : str = 'x'):

        ''' GPAW superclass for interfacing with QEH
        building block calculatotions.

        Parameters
        ----------
        df : DielectricFunction
            the dielectric function calculator
        qinf_min : float
            the lower limit of the qinf grid as a fraction
            of the lowest non inf q-point
        qinf_max : float
            the upper limit of the qinf grid as a fraction
            of the highest non inf q-point
        nq_inf : int
            the number of qinf grid points
        direction : str (either 'x' or 'y')
            the direction of the q-grid
        '''

        self.df = df # DielectricFunctionCalculator
        self.L = df.gs.gd.cell_cv[2, 2]
        self.omega_w = self.df.chi0calc.wd.omega_w
        self.direction = direction
        self.context = self.df.context
        self.qinf_min = qinf_min
        self.qinf_max = qinf_max
        self.nq_inf = nq_inf
        
        super().__init__()
        
    def get_q_grid(self, q_max : float | None = None):
        # First get q-points on the grid
        qdim = {'x': 0, 'y': 1}
        qdir = qdim[self.direction]
        kd = self.df.gs.kd
        Nk = kd.N_c[qdir]
        gd = self.df.gs.gd
        icell_cv = gd.icell_cv
        
        q_qc=np.zeros([Nk//2+1, 3], dtype=float)
        q_qc[:, qdir] = np.linspace(0, 0.5, Nk//2+1)
        q_qc = q_qc[1:-1] # exclude gamma
        q_qv = q_qc @ icell_cv * 2 * pi

        if q_max is not None:
            q_mask = np.linalg.norm(q_qv, axis=1) <= q_max
            q_qc = q_qc[q_mask]
            q_qv = q_qv[q_mask]

        # get additional q-points around gamma
        qinf_qc = np.zeros([self.nq_inf, 3], dtype=float)
        qinf_qc[:, qdir] = np.linspace(self.qinf_min*q_qc[0, qdir],
                                       self.qinf_max*q_qc[0, qdir],
                                       self.nq_inf)
        qinf_qv  = qinf_qc @ icell_cv * 2 * pi

        # make list of QPoints for calculation
        Q_q = [QPoint(q_c=q_c, q_v=q_v, interpolate_from_gamma=True)
               for q_c, q_v in zip(qinf_qc, qinf_qv)]
        Q_q.extend([QPoint(q_c=q_c, q_v=q_v) for q_c, q_v
                    in zip(q_qc, q_qv)])

        return Q_q

    def get_z_grid(self):
        r = self.df.gs.gd.get_grid_point_coordinates()
        return r[2, 0, 0, :]
        
    def get_chi_wGG(self, qpoint : QPoint):
        if qpoint.interpolate_from_gamma:
            chi0_dyson_eqs = self.df.get_chi0_dyson_eqs([0, 0, 0],
                                                        truncation='2D')
            qpd, chi_wGG, wblocks = chi0_dyson_eqs.rpa_density_response(
                qinf_v=qpoint.q_v, direction=self.direction)
        else:
            chi0_dyson_eqs = self.df.get_chi0_dyson_eqs(qpoint.q_c,
                                                        truncation='2D')
            qpd, chi_wGG, wblocks = chi0_dyson_eqs.rpa_density_response()

        chi_wGG = wblocks.gather(chi_wGG, root=0)
        G_Gv = qpd.get_reciprocal_vectors(add_q=False)

        return chi_wGG, G_Gv
