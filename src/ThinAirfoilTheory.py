import numpy as np

from GeometryHandler import GeometryHandler
from CamberLineFitter import CamberLineFitter
from AnalyticalSolver import AnalyticalSolver
from NumericalPanelSolver import NumericalPanelSolver

class ThinAirfoilTheory:
    def __init__(self, filename):
        self.filename = filename
        self.geometry = None
        self.camber_poly = None
        self.analytical_solver = None
        self.numerical_solver = None

    def set_flow_conditions(self, V_inf, rho, alpha_deg):
        self.V_inf = V_inf
        self.rho = rho
        self.alpha_rad = np.radians(alpha_deg)

    def set_N_panels(self, N):
        self.N = N

    def prepare(self):
        self.geometry = GeometryHandler(self.filename)
        x, y = self.geometry.read_surface_coordinates()

        fitter = CamberLineFitter(x, y)
        _, _, poly = fitter.compute_mean_camber()
        self.camber_poly = poly

        self.analytical_solver = AnalyticalSolver(poly)
        self.numerical_solver = NumericalPanelSolver(poly)

    def run_analytical(self, alpha_deg):
        alpha_rad = np.radians(alpha_deg)
        Cl, Cm = self.analytical_solver.solve(alpha_rad)
        print(f"[Analytical] Cl = {Cl:.4f}, Cm_LE = {Cm:.4f}")
        return Cl, Cm

    def run_numerical(self, alpha_deg):
        Cl, gamma, x_quarter = self.numerical_solver.run(self.N, self.V_inf, self.rho, alpha_deg)
        print(f"[Numerical] Cl = {Cl:.4f}")
        return Cl, gamma, x_quarter
