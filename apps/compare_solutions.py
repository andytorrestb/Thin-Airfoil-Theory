import apps_header

from pathlib import Path
import numpy as np

from ThinAirfoilTheory import ThinAirfoilTheory

here = Path(__file__).resolve().parent
data_file = str((here / '..' / 'foils' / 'NACA0012.dat').resolve())

theory = ThinAirfoilTheory(data_file)
theory.prepare()
theory.set_N_panels(N=100)
theory.set_flow_conditions(V_inf=50.0, rho=1.225, alpha_deg=5.0)

alpha = np.linspace(-4, 15, 50)  # Angle of attack from -4 to 15 degrees


Cl_numerical = []
Cm_le_numerical = []

Cl_analytical = []
Cm_le_analytical = []

for a in alpha:
    # Numerical solver returns (Cl, gamma(x), x_quarter)
    Cl_num, gamma, x_quarter = theory.run_numerical(alpha_deg=a)
    # Analytical solver returns (Cl, Cm_le)
    Cl_an, Cm_le_an = theory.run_analytical(alpha_deg=a)

    # Compute numerical Cm_le from the bound vortex distribution gamma(x)
    # Using thin airfoil relation: M_LE' = - rho * V_inf * ∫ x * gamma(x) dx
    # Non-dimensional coefficient (c=1): Cm_le = 2 * M_LE' / (rho * V_inf^2)
    dx = 1.0 / theory.N
    M_le_prime = -theory.rho * theory.V_inf * np.sum(x_quarter * gamma) * dx
    Cm_le_num = 2 * M_le_prime / (theory.rho * theory.V_inf**2)

    Cl_numerical.append(Cl_num)
    Cm_le_numerical.append(Cm_le_num)

    Cl_analytical.append(Cl_an)
    Cm_le_analytical.append(Cm_le_an)


Cl_numerical = np.array(Cl_numerical)
Cm_le_numerical = np.array(Cm_le_numerical)
Cl_analytical = np.array(Cl_analytical)
Cm_le_analytical = np.array(Cm_le_analytical)
alpha = np.array(alpha)

print("Shapes:")
print("alpha:", alpha.shape)
print("Cl_numerical:", Cl_numerical.shape)
print("Cm_le_numerical:", Cm_le_numerical.shape)
print("Cl_analytical:", Cl_analytical.shape)
print("Cm_le_analytical:", Cm_le_analytical.shape)


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(alpha, Cl_numerical, label='Numerical Solver', linestyle='--')
plt.plot(alpha, Cl_analytical, label='Analytical Solver', linestyle='-')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Lift Coefficient (Cl)')
plt.title('Lift Coefficient Comparison')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(alpha, Cm_le_numerical, label='Numerical Solver', linestyle='--')
plt.plot(alpha, Cm_le_analytical, label='Analytical Solver', linestyle='-')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Moment Coefficient about Leading Edge (Cm_le)')
plt.title('Moment Coefficient Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('comparison_numerical_analytical.png', dpi=300)