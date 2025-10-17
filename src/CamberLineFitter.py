import numpy as np
from scipy.spatial import Voronoi

class CamberLineFitter:
    def __init__(self, x_surf, y_surf):
        self.x_surf = x_surf
        self.y_surf = y_surf
        self.poly = None

    def compute_mean_camber(self, degree=5):
        points = np.column_stack((self.x_surf, self.y_surf))
        vor = Voronoi(points)
        x_mean, y_mean = [], []

        for vx, vy in vor.vertices:
            if 0 <= vx <= 1 and -0.3 <= vy <= 0.3:
                x_mean.append(vx)
                y_mean.append(vy)

        x_mean = np.array(x_mean)
        y_mean = np.array(y_mean)

        sort_idx = x_mean.argsort()
        x_mean = x_mean[sort_idx]
        y_mean = y_mean[sort_idx]

        self.poly = np.poly1d(np.polyfit(x_mean, y_mean, degree))
        return x_mean, y_mean, self.poly
