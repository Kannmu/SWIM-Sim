import numpy as np
from scipy.stats.qmc import PoissonDisk

class ReceptorArray:
    def __init__(self, roi_size_mm, n_receptors, min_dist_mm, seed=None):
        self.roi_size = roi_size_mm
        self.n_receptors = n_receptors
        self.min_dist = min_dist_mm
        self.seed = seed
        self.coordinates = None

    def generate(self):
        """
        Generates receptor coordinates using Poisson-disk sampling.
        Returns coordinates in mm centered at (0,0).
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Calculate radius in normalized coordinates [0, 1]
        # ROI is square: roi_size x roi_size
        radius = self.min_dist / self.roi_size
        
        # Initialize PoissonDisk sampler
        engine = PoissonDisk(d=2, radius=radius, seed=self.seed)
        
        # Generate samples
        samples = engine.fill_space()
        
        # Check count and retry with relaxed radius if necessary
        retries = 0
        current_radius = radius
        
        while len(samples) < self.n_receptors and retries < 10:
            retries += 1
            current_radius *= 0.95
            print(f"Warning: PoissonDisk generated only {len(samples)} points. Retrying with radius {current_radius:.4f} (Try {retries}/10)")
            engine = PoissonDisk(d=2, radius=current_radius, seed=self.seed)
            samples = engine.fill_space()

        # Check count
        if len(samples) < self.n_receptors:
            print(f"Warning: PoissonDisk generated only {len(samples)} points, fewer than requested {self.n_receptors}. Reducing radius slightly or accepting fewer.")
            pass
        
        if len(samples) > self.n_receptors:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(samples), self.n_receptors, replace=False)
            samples = samples[indices]
            
        self.coordinates = (samples - 0.5) * self.roi_size
        return self.coordinates

    def get_coordinates(self):
        if self.coordinates is None:
            self.generate()
        return self.coordinates
