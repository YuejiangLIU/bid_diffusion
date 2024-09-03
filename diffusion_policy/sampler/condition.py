import numpy as np
import pdb

# def compound_noise(pred, k):
#     BS, T, D = pred.shape
#     initial_noise = np.random.randn(BS, T-1, D)
#     pdb.set_trace()
#     step_noise = (pred[:, 1:] - pred[:, :-1]) * initial_noise
#     sequence_noise = step_noise
#     return sequence_noise

# def exponential_noise(shape, k):
#     BS, T, D = shape  # Unpacking the shape tuple
#     initial_noise = np.random.randn(BS, 1, D)
#     norms = np.linalg.norm(initial_noise, axis=2, keepdims=True)
#     normalized_initial_noise = initial_noise / norms * k
#     time_scale = np.linspace(0, 1, num=T).reshape(1, -1, 1)
#     exponential_scale = time_scale ** k  # Squaring the linear scale to make it quadratic
#     growing_noise = normalized_initial_noise  * exponential_scale
#     return growing_noise

# def constant_noise(shape, k):
#     BS, T, D = shape  # Unpacking the shape tuple
#     initial_noise = np.random.randn(BS, 1, D)
#     norms = np.linalg.norm(initial_noise, axis=2, keepdims=True)
#     normalized_initial_noise = initial_noise / norms * k
#     constant_scale = np.ones((1, T, 1))  # Same constant value for all time steps
#     sequence_noise = normalized_initial_noise * constant_scale
#     return sequence_noise

# def linear_noise(shape):
#     BS, T, D = shape  # Unpacking the shape tuple
#     initial_noise = np.random.randn(BS, 1, D)
#     growing_noise = initial_noise * np.linspace(0, 1, num=T).reshape(1, -1, 1)
#     return growing_noise

# def quadratic_noise(shape):
#     BS, T, D = shape  # Unpacking the shape tuple
#     initial_noise = np.random.randn(BS, 1, D)
#     norms = np.linalg.norm(initial_noise, axis=2, keepdims=True)
#     normalized_initial_noise = initial_noise / norms
#     time_scale = np.linspace(0, 1, num=T).reshape(1, -1, 1)
#     quadratic_scale = time_scale ** 2  # Squaring the linear scale to make it quadratic
#     growing_noise = normalized_initial_noise  * quadratic_scale
#     return growing_noise


class NoiseGenerator:
    def __init__(self, noise_strength, correlation_factor=0.9):
        self.noise_strength = noise_strength
        self.correlation_factor = correlation_factor
        self.previous_noise = None

    def step(self, pred):
        # Generate random noise
        # noise_seed = np.random.randn(*pred) * self.noise_strength
        noise_seed = (np.random.rand(pred.shape[0], 1, pred.shape[2]) + 0.5) * np.random.choice([-1, 1], size=(pred.shape[0], 1, pred.shape[2]))
        action_step = (pred[:, 1:] - pred[:, :-1])
        noise_step = noise_seed.repeat(action_step.shape[1], axis=1) * action_step * self.noise_strength

        # If it's the first time step, there's no previous noise, so use the seed directly
        if self.previous_noise is None:
            self.previous_noise = noise_step
        else:
            # Combine the previous noise with new noise to create temporally correlated noise
            noise_step = self.correlation_factor * self.previous_noise + (1 - self.correlation_factor) * noise_step
            self.previous_noise = noise_step

        noise_cum = np.cumsum(noise_step, axis=1)
        # print('noise_cum', noise_cum)
        # pdb.set_trace()

        return noise_cum
