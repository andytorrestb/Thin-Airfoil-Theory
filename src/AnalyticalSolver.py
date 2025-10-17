import sympy as sp
import numpy as np
import scipy.integrate as integrate


class AnalyticalSolver:
    """Thin Airfoil Theory analytical solver.

    Computes the A coefficients from the derivative of the camber line polynomial
    and derives lift and moment coefficients, zero-lift angle, and center of
    pressure similarly to the original root `ThinAirfoilTheory.py` implementation.
    """

    def __init__(self, camber_poly):
        self.poly = camber_poly

        # Results populated after solve()
        self.coefficients = None
        self.cl = None
        self.zero_lift_angle = None
        self.cm_le = None
        self.cm_quarter = None
        self.x_cp = None
        self.lambda_der = None

    def solve(self, alpha_rad, chord=1.0, n_coeff=3, report=False):
        """Solve thin airfoil theory for given angle (radians).

        Returns (Cl, Cm_LE) to preserve the previous API, and also fills
        attributes with additional derived data matching the original script.
        """
        # Ensure reasonable number of coefficients
        assert n_coeff >= 2, 'More than 1 coefficient should be computed in order to derive data from this theory'

        # derivative of the camber polynomial (numpy.poly1d)
        poly_der = self.poly.deriv(1)

        # Build sympy polynomial from derivative coefficients and perform change of variable
        x = sp.Symbol('x')
        theta = sp.Symbol('theta')
        poly_expr = sp.Poly(poly_der.coefficients, x)
        subs_expr = poly_expr.subs({x: 0.5 * chord * (1 - sp.cos(theta))})
        lam = sp.lambdify(theta, subs_expr.as_expr(), modules='numpy')
        self.lambda_der = lam

        # Compute A coefficients
        A = []
        A0 = alpha_rad - (1 / np.pi) * integrate.quad(lam, 0, np.pi)[0]
        A.append(A0)
        for i in range(1, n_coeff):
            Ai = (2 / np.pi) * integrate.quad(lambda t: lam(t) * np.cos(i * t), 0, np.pi)[0]
            A.append(Ai)

        self.coefficients = A

        # Compute derived data (lift, moments, zero-lift angle, center of pressure)
        self._compute_relevant_data(A, chord)

        if report:
            print("\n-----------------------------------------------------------------------------------\n")
            print(f"Angle of attack: {alpha_rad*180/np.pi:.2f} degrees.")
            print(f"Zero lift angle of attack: {self.zero_lift_angle*180/np.pi:.2f} degrees.")
            print(f"Lift coefficient: {self.cl:.5f}")
            print(f"Moment coefficient around the leading edge: {self.cm_le:.5f}")
            print(f"Moment coefficient around the quarter chord: {self.cm_quarter:.5f}")
            print("\n-----------------------------------------------------------------------------------\n")

        # Preserve original return values
        Cl = self.cl
        Cm = self.cm_le
        return Cl, Cm

    def _compute_relevant_data(self, coefficients, chord):
        # Cl
        self.cl = 2 * np.pi * (coefficients[0] + 0.5 * coefficients[1])

        # Zero-lift angle
        factor = -(1 / np.pi)
        self.zero_lift_angle = factor * integrate.quad(lambda angle: self.lambda_der(angle) * (np.cos(angle) - 1), 0, np.pi)[0]

        # Moments
        # Leading edge moment
        # guard accesses for coefficients beyond available indices
        a1 = coefficients[1] if len(coefficients) > 1 else 0.0
        a2 = coefficients[2] if len(coefficients) > 2 else 0.0
        self.cm_le = -(self.cl / 4 + np.pi / 4 * (a1 - a2))

        # Quarter-chord moment
        self.cm_quarter = np.pi / 4 * (a2 - a1)

        # Center of pressure
        # avoid division by zero
        if self.cl != 0:
            self.x_cp = chord / 4 * (1 + np.pi / self.cl * (a1 - a2))
        else:
            self.x_cp = None
