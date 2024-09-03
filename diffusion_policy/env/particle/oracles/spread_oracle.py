import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts


class SpreadOracle(py_policy.PyPolicy):
    def __init__(self, env):
        super(SpreadOracle, self).__init__(
            env.time_step_spec(), env.action_spec()
        )
        self._env = env

    def _action(self, time_step, policy_state=()):
        obs = time_step.observation
        # goal = self._env.world.landmarks[1 - self._env.other_agent_idx].state.p_pos
        other_dist1 = np.linalg.norm(self._env.world.landmarks[0].state.p_pos - obs[2:4])
        other_dist2 = np.linalg.norm(self._env.world.landmarks[1].state.p_pos - obs[2:4])
        ego_dist1 = np.linalg.norm(self._env.world.landmarks[0].state.p_pos - obs[:2])
        ego_dist2 = np.linalg.norm(self._env.world.landmarks[1].state.p_pos - obs[:2])
        if ego_dist1 < other_dist1 - 0.1:
            # print("Ego agent: moving towards goal 1")
            goal = self._env.world.landmarks[0].state.p_pos
        elif ego_dist2 < other_dist2 - 0.1:
            # print("Ego agent: moving towards goal 0")
            goal = self._env.world.landmarks[1].state.p_pos
        else:
            goal = self._env.world.landmarks[1 - self._env.other_agent_idx].state.p_pos
        action = goal - obs[:2]

        # Avoid collision by moving away from the other agent
        other_agent_pos = obs[2:4]
        delta_pos = other_agent_pos - obs[:2]
        if np.linalg.norm(delta_pos) < 0.4:
            # Move away from agent while moving towards goal
            # print("Ego agent: too close to other agent")
            action = 0.1 * action - 2 * delta_pos

        # Ensure action norm is less than 1 and at least 0.25
        if np.linalg.norm(action) > 1:
            action /= np.linalg.norm(action)
        elif np.linalg.norm(action) < 0.5:
            action = action / np.linalg.norm(action) * 0.5

        action += np.random.normal(0, 0.1, size=2)

        if np.linalg.norm(action) > 1:
            action /= np.linalg.norm(action)
        elif np.linalg.norm(action) < 0.5:
            action = action / np.linalg.norm(action) * 0.5

        return policy_step.PolicyStep(action, policy_state)

    def reset(self):
        pass
