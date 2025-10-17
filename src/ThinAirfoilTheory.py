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

    def run_numerical(self, N, V_inf, rho):
        Cl, gamma = self.numerical_solver.run(N, V_inf, rho)
        print(f"[Numerical] Cl = {Cl:.4f}")
        return Cl, gamma
