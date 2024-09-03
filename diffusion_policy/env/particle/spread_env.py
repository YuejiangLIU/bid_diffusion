from functools import partial
import gym
from gym import spaces
import numpy as np

from diffusion_policy.env.particle.core import Agent, World, Landmark, Action


def go_to_landmark(agent, world, landmark_idx, rng, avoid=True):
    landmark = world.landmarks[landmark_idx]
    u = landmark.state.p_pos - agent.state.p_pos

    # Avoid collision by moving away from the other agent
    if avoid:
        other_agent_pos = world.agents[0].state.p_pos
        delta_pos = other_agent_pos - agent.state.p_pos
        if np.linalg.norm(delta_pos) < 0.5:
            # Move away from agent while moving towards goal
            # print("Other agent: too close to other agent")
            u = u - 2 * delta_pos

    # Ensure action norm is less than 0.9 and at least 0.5
    if np.linalg.norm(u) > 0.9:
        u = u / np.linalg.norm(u) * 0.9
    elif np.linalg.norm(u) < 0.5:
        u = u / np.linalg.norm(u) * 0.5

    # Add noise to action
    u += rng.normal(0, 0.1, size=2)

    # Renormalize action
    if np.linalg.norm(u) > 0.9:
        u = u / np.linalg.norm(u) * 0.9
    elif np.linalg.norm(u) < 0.5:
        u = u / np.linalg.norm(u) * 0.5

    action = Action()
    action.u = u
    return action


class ParticleSpreadEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
        render_action=True,
        render_size=96,
        success_threshold=0.075,
        adversarial=True,
    ):
        self._seed = None
        self.seed()
        self.render_action = render_action
        self.render_size = render_size
        self.success_threshold = success_threshold
        self.adversarial = adversarial

        self.world = self.make_world()

        # agent_pos, other_agent_pos
        self.observation_space = spaces.Box(
            low=np.array([-np.inf,] * 8, dtype=np.float64),
            high=np.array([np.inf,] * 8, dtype=np.float64),
            shape=(8,),
            dtype=np.float64,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.world.agents[0].u_range] * 2, dtype=np.float64),
            high=np.array([self.world.agents[0].u_range] * 2, dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.viewers = [None]

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        # for i, agent in enumerate(world.agents):
        #     agent.color = np.array([0.35, 0.35, 0.85])
        world.agents[0].color = np.array([0.35, 0.35, 0.85])
        world.agents[1].color = np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self.np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # reset landmark positions if they are too close
        while np.linalg.norm(world.landmarks[0].state.p_pos - world.landmarks[1].state.p_pos) < 0.8:
            # print("Resetting landmark positions")
            world.landmarks[1].state.p_pos = self.np_random.uniform(-1, +1, world.dim_p)

        for agent in world.agents:
            agent.state.p_pos = self.np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        if self.adversarial:
            landmark_midpoint = (world.landmarks[0].state.p_pos + world.landmarks[1].state.p_pos) / 2
            world.agents[1].state.p_pos = landmark_midpoint + self.np_random.uniform(-0.1, 0.1, world.dim_p)
        # reset agent positions if they are too close or if they are too close to landmarks
        while np.linalg.norm(world.agents[0].state.p_pos - world.agents[1].state.p_pos) < 0.5 or \
                np.linalg.norm(world.agents[0].state.p_pos - world.landmarks[0].state.p_pos) < 0.8 or \
                np.linalg.norm(world.agents[0].state.p_pos - world.landmarks[1].state.p_pos) < 0.8:
            world.agents[0].state.p_pos = self.np_random.uniform(-1, +1, world.dim_p)

        if self.adversarial:
            steps = 0
            self.switch_times = []
            while steps <= 350:
                switch_time = self.np_random.integers(4, 8)
                self.switch_times.append(switch_time)
                steps += switch_time

            self.current_switch = 0
            self.steps_since_last_switch = 0

            self.other_agent_idx = self.np_random.integers(0, len(world.landmarks))
            world.agents[1].action_callback = partial(
                self.go_to_landmark_adversarial, rng=self.np_random)
        else:
            # set other agent's intent
            self.other_agent_idx = self.np_random.integers(0, len(world.landmarks))
            world.agents[1].action_callback = partial(
                go_to_landmark, landmark_idx=self.other_agent_idx, rng=self.np_random)

    def go_to_landmark_adversarial(self, agent, world, rng):
        self.steps_since_last_switch += 1
        if self.steps_since_last_switch < self.switch_times[self.current_switch]:
            return go_to_landmark(agent, world, self.other_agent_idx, rng, avoid=False)

        self.current_switch += 1
        self.steps_since_last_switch = 0

        # Check which landmark is closer to the other agent to determine which landmark to go to
        # agent_pos = world.agents[0].state.p_pos
        # dist1 = np.linalg.norm(agent_pos - world.landmarks[0].state.p_pos)
        # dist2 = np.linalg.norm(agent_pos - world.landmarks[1].state.p_pos)
        # if dist1 < dist2:
        #     landmark_idx = 0
        # else:
        #     landmark_idx = 1

        # Check the direction of agent velocity vector to determine which landmark to go to
        agent_vel = world.agents[0].state.p_vel
        # Get cosine similarity between agent velocity and vector to each landmark
        cos_sim1 = np.dot(agent_vel, world.landmarks[0].state.p_pos - world.agents[0].state.p_pos) / \
            (np.linalg.norm(agent_vel) * np.linalg.norm(world.landmarks[0].state.p_pos - world.agents[0].state.p_pos))
        cos_sim2 = np.dot(agent_vel, world.landmarks[1].state.p_pos - world.agents[0].state.p_pos) / \
            (np.linalg.norm(agent_vel) * np.linalg.norm(world.landmarks[1].state.p_pos - world.agents[0].state.p_pos))

        if cos_sim1 > cos_sim2:
            landmark_idx = 0
        else:
            landmark_idx = 1
        self.other_agent_idx = landmark_idx
        return go_to_landmark(agent, world, landmark_idx, rng, avoid=False)

    def reset(self):
        self.reset_world(self.world)
        self._reset_render()
        return self._get_obs()

    def step(self, action):
        # set action for agent 0
        self.world.agents[0].action.u = action

        self.world.step()
        if self.adversarial:
            success = self.is_success_adversarial()
        else:
            success = self.is_success_collaborative()
        collision = self.is_collision()

        if success:
            reward = 1.0
        elif collision:
            reward = -1.0
        else:
            reward = 0.0

        done = collision or success

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def is_collision(self):
        agent_pos = self.world.agents[0].state.p_pos
        other_agent_pos = self.world.agents[1].state.p_pos

        delta_pos = agent_pos - other_agent_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        collision_threshold = self.world.agents[0].size * 2
        return dist < collision_threshold

    def is_success_adversarial(self):
        # check if ego agent is covering a landmark
        agent_pos = self.world.agents[0].state.p_pos
        for landmark in self.world.landmarks:
            delta_pos = agent_pos - landmark.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            if dist <= self.success_threshold:
                return True
        return False

    def is_success_collaborative(self):
        # check if each landmark is covered by an agent
        agent_pos = self.world.agents[0].state.p_pos
        other_agent_pos = self.world.agents[1].state.p_pos

        for landmark in self.world.landmarks:
            min_dist = np.inf
            for pos in [agent_pos, other_agent_pos]:
                delta_pos = pos - landmark.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_pos)))
                min_dist = min(min_dist, dist)
            if min_dist > self.success_threshold:
                return False
        return True

    def _get_info(self):
        info = {
            "agent_pos": np.array(self.world.agents[0].state.p_pos),
            "other_agent_pos": np.array(self.world.agents[1].state.p_pos),
            # TODO(anxie): add information about other agent's intent
        }
        return info

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _get_obs(self):
        obs = np.array(
            tuple(self.world.agents[0].state.p_pos) \
            + tuple(self.world.agents[1].state.p_pos) \
            + tuple(self.world.landmarks[0].state.p_pos) \
            + tuple(self.world.landmarks[1].state.p_pos))
        return obs

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from diffusion_policy.env.particle import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from diffusion_policy.env.particle import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from diffusion_policy.env.particle import rendering
            # update bounds to center around agent
            cam_range = 1
            pos = np.zeros(self.world.dim_p)
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results[0]


if __name__ == "__main__":
    env = ParticleSpreadEnv()
    obs = env.reset()
    ims = []
    for _ in range(100):
        action = env.world.landmarks[env.other_agent_idx].state.p_pos - obs[:2]
        if np.linalg.norm(action) > 1:
            action /= np.linalg.norm(action)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        im = env.render(mode='rgb_array')
        ims.append(im)
    env.close()

    import imageio
    imageio.mimsave('particle_spread.gif', ims, fps=10)
