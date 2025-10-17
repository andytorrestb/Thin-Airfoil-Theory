import unittest
from pathlib import Path
import numpy as np

import test_header
from ThinAirfoilTheory import ThinAirfoilTheory

class TestThinAirfoilTheory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate the foil data file relative to this test file
        here = Path(__file__).resolve().parent
        data_file = str((here / '..' / 'foils' / 'NACA0012.dat').resolve())

        cls.theory = ThinAirfoilTheory(data_file)
        cls.theory.prepare()

    def test_analytical_solver(self):
        # Run analytical solver and verify returned values
        Cl, Cm_le = self.theory.run_analytical(alpha_deg=5)
        self.assertIsInstance(Cl, float)
        self.assertTrue(np.isfinite(Cl))
        self.assertIsInstance(Cm_le, float)
        self.assertTrue(np.isfinite(Cm_le))
        self.assertAlmostEqual(Cl, 0.5483, places=2)
        self.assertAlmostEqual(Cm_le, -0.13708, places=2)