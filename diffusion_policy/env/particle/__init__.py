"""Environments using kitchen and Franka robot."""
from gym.envs.registration import register


register(
    id="particle-spread-v0",
    entry_point="diffusion_policy.env.particle.spread_env:ParticleSpreadEnv",
    max_episode_steps=200,
    reward_threshold=1.0,
)
