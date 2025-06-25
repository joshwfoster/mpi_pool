import numpy as np
from mpi_pool import MPIPool

# Define evaluator
class Evaluator:
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, x):
        return self.scale * np.sum(x**2)

# Optional transform: normalize each input vector
def transform_fn(xs):
    return xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)

if __name__ == "__main__":
    # Instantiate the pool — worker processes will block and exit internally
    pool = MPIPool(
        evaluator_cls=Evaluator,
        evaluator_kwargs={"scale": 2.0},
        transform_fn=transform_fn
    )

    # Only master process reaches this point
    xs = np.random.randn(256, 8)
    results = pool.map_array(xs)

    # Print some results
    for i in range(5):
        print(f"Input {i}: {xs[i]}")
        print(f"Result {i}: {results[i]:.4f}\n")

    pool.close()  # Send shutdown signal to workers
