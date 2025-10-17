import os
import numpy as np

class GeometryHandler:
    def __init__(self, filename):
        self.filename = filename
        self.x_coords = None
        self.y_coords = None
        self.header = None

    def read_surface_coordinates(self):
        with open(self.filename, 'r') as file:
            lines = file.readlines()

        self.header = lines[0].strip()
        x_vals, y_vals = [], []

        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    x_vals.append(x)
                    y_vals.append(y)
                except ValueError:
                    continue

        self.x_coords = np.array(x_vals)
        self.y_coords = np.array(y_vals)
        return self.x_coords, self.y_coords
