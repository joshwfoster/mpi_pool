import numpy as np
from mpi_pool import MPIPool  # assumes your MPIPool code is saved as mpi_pool.py

class Evaluator:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, x):
        # Simulate work: squared norm scaled by `self.scale`
        return self.scale * np.sum(np.square(x))


def transform_fn(xs):
    # Normalize each input vector
    norms = np.linalg.norm(xs, axis=1, keepdims=True)
    return xs / (norms + 1e-8)


if __name__ == '__main__':
    # Only master will run this block fully
    with MPIPool(evaluator_cls=Evaluator, evaluator_kwargs={'scale': 2.0}, transform_fn=transform_fn) as pool:
        if pool.is_master():
            # Generate random input: 256 points in 8 dimensions
            xs = np.random.randn(256, 8)
            results = pool.map_array(xs)

            # Print a few outputs
            for i, (x, y) in enumerate(zip(xs, results)):
                if i >= 5:
                    break
                print(f"Input {i}: {x}")
                print(f"Transformed and evaluated result: {y:.4f}\n")
