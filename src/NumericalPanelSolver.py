import numpy as np

class NumericalPanelSolver:
    """
    Implements the vortex panel method to approximate thin airfoil theory numerically.
    """
    def __init__(self, camber_poly):
        """
        :param camber_poly: A NumPy polynomial object representing the camber line z(x)
        """
        self.poly = camber_poly

    def dz_dx(self, x):
        """
        Returns the slope of the camber line at point x.
        """
        return np.polyder(self.poly)(x)

    def run(self, N, V_inf, rho, alpha_deg=0.0):
        """
        Executes the numerical thin airfoil theory using vortex panels.
        
        :param N: Number of panels
        :param V_inf: Freestream velocity (m/s)
        :param rho: Freestream density (kg/m³)
        :param alpha_deg: Angle of attack in degrees
        :return: Lift coefficient (Cl), circulation distribution (gamma), x-panel locations
        """
        alpha_rad = np.radians(alpha_deg)
        dx = 1.0 / N

        # Discretize the chord
        x_quarter = np.linspace(dx / 4, 1 - dx / 4, N)         # vortex points (¼-chord)
        x_three_quarter = np.linspace(3 * dx / 4, 1 - dx / 4, N)  # control points (¾-chord)

        # Build influence matrix A
        A = np.zeros((N, N))
        for i in range(N):
            xi = x_three_quarter[i]
            for j in range(N):
                xj = x_quarter[j]
                if i == j:
                    A[i, j] = 0.5  # self-induced velocity for flat panels
                else:
                    A[i, j] = (1 / (2 * np.pi)) * (xi - xj) / ((xi - xj) ** 2)

        # Build RHS vector b: includes camber slope + angle of attack
        b = -V_inf * np.array([self.dz_dx(xi) + alpha_rad for xi in x_three_quarter])

        # Solve for vortex strengths γ
        gamma = np.linalg.solve(A, b)

        # Compute lift coefficient
        circulation = np.sum(gamma) * dx
        L_prime = rho * V_inf * circulation
        Cl = 2 * L_prime / (rho * V_inf ** 2)

        return Cl, gamma, x_quarter
