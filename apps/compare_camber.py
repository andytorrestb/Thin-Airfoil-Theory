import apps_header

from pathlib import Path
import numpy as np

from ThinAirfoilTheory import ThinAirfoilTheory

here = Path(__file__).resolve().parent
naca0012_path = str((here / '..' / 'foils' / 'NACA0012.dat').resolve())
naca2012_path = str((here / '..' / 'foils' / 'NACA2112.dat').resolve())

theory_0012 = ThinAirfoilTheory(naca0012_path)
theory_0012.prepare()
theory_0012.set_N_panels(N=10)
theory_0012.set_flow_conditions(V_inf=90.0, rho=1.225, alpha_deg=5.0)

theory_2012 = ThinAirfoilTheory(naca2012_path)
theory_2012.prepare()
theory_2012.set_N_panels(N=10)
theory_2012.set_flow_conditions(V_inf=90.0, rho=1.225, alpha_deg=5.0)
alpha = 5.0  # Angle of attack in degrees

Cl_0012, gamma_0012, x_quarter_0012 = theory_0012.run_numerical(alpha_deg=alpha)
Cl_2012, gamma_2012, x_quarter_2012 = theory_2012.run_numerical(alpha_deg=alpha)

print(f"NACA 0012: Cl = {Cl_0012:.4f}")
print(f"NACA 2012: Cl = {Cl_2012:.4f}")