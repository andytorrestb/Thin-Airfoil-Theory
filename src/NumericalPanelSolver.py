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
        # Use poly1d.deriv() to avoid ambiguity with np.polyder on different polynomial types
        return self.poly.deriv(1)(x)

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
        x_quarter = np.linspace(dx / 4, 1 - dx / 4, N)              # vortex points (¼-chord)
        x_three_quarter = np.linspace(3 * dx / 4, 1 - 3 * dx / 4, N) # control points (¾-chord)

        # Build influence matrix A (discretization of the Cauchy principal value integral)
        # 0.5*gamma(x_i) + (1/(2π)) * PV ∫ gamma(ξ)/(x_i-ξ) dξ = V_inf*(alpha - dz/dx)
        # Simple rectangle rule: A[i,j] = dx/(2π) * 1/(x_i - x_j) for i≠j; A[i,i] = 0.5
        A = np.zeros((N, N))
        for i in range(N):
            xi = x_three_quarter[i]
            for j in range(N):
                xj = x_quarter[j]
                if i == j:
                    A[i, j] = 0.5
                else:
                    A[i, j] = (dx / (2 * np.pi)) * (1.0 / (xi - xj))

        # Build RHS vector b: V_inf * (alpha - dz/dx)
        b = V_inf * np.array([alpha_rad - self.dz_dx(xi) for xi in x_three_quarter])

        # Solve for vortex sheet strength γ(x_j)
        gamma = np.linalg.solve(A, b)

        # Compute lift coefficient: L' = ρ V_inf ∫ γ(x) dx ≈ ρ V_inf * (Σ γ_j) dx
        circulation = np.sum(gamma) * dx
        L_prime = rho * V_inf * circulation
        Cl = 2 * L_prime / (rho * V_inf ** 2)

        return Cl, gamma, x_quarter
