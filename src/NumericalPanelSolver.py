import numpy as np

class NumericalPanelSolver:
    def __init__(self, camber_poly):
        self.poly = camber_poly

    def dz_dx(self, x):
        return np.polyder(self.poly)(x)

    def run(self, N, V_inf, rho):
        dx = 1.0 / N
        x_quarter = np.linspace(dx / 4, 1 - dx / 4, N)
        x_three_quarter = np.linspace(3 * dx / 4, 1 - dx / 4, N)

        A = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                xi = x_three_quarter[i]
                xj = x_quarter[j]
                if i == j:
                    A[i, j] = 0.5
                else:
                    A[i, j] = (1 / (2 * np.pi)) * (xi - xj) / ((xi - xj) ** 2)

        b = -V_inf * np.array([self.dz_dx(xi) for xi in x_three_quarter])
        gamma = np.linalg.solve(A, b)

        circulation = np.sum(gamma) * dx
        L_prime = rho * V_inf * circulation
        Cl = 2 * L_prime / (rho * V_inf**2)

        return Cl, gamma
