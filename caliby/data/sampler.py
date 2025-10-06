import numpy as np


class Sampler:
    def __init__(self, weights: np.ndarray):
        self.weights = weights / weights.sum()

    def sample(self, rng: np.random.Generator):
        """
        Sample indices from the dataset infinitely.

        Args:
            random: Random state to use for sampling.

        Returns:
            Iterator[int]: Iterator of sampled indices.
        """
        while True:
            # O(n) per draw
            idx = rng.choice(len(self.weights), p=self.weights)
            yield idx
