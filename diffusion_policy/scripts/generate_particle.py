if __name__ == "__main__":
    import sys
    import os
    import pathlib
    import imageio

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import click
import pathlib
import numpy as np
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories.time_step import StepType
from diffusion_policy.env.particle.spread_env import ParticleSpreadEnv
from diffusion_policy.env.particle.oracles.spread_oracle import SpreadOracle

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-n', '--n_episodes', default=1000)
@click.option('-c', '--chunk_length', default=-1)
def main(output, n_episodes, chunk_length):

    buffer = ReplayBuffer.create_empty_numpy()
    env = TimeLimit(GymWrapper(ParticleSpreadEnv()), duration=350)
    # for i in tqdm(range(n_episodes)):
    i = 0
    num_failed_episodes = 0
    while buffer.n_episodes < n_episodes:
        print(i)
        obs_history = list()
        action_history = list()

        env.seed(i)
        policy = SpreadOracle(env)
        time_step = env.reset()
        policy_state = policy.get_initial_state(1)
        # images = [env.render(mode='rgb_array')]
        while True:
            action_step = policy.action(time_step, policy_state)
            obs = time_step.observation
            action = action_step.action
            obs_history.append(obs)
            action_history.append(action)

            if time_step.step_type == 2:
                # assert time_step.reward == 1.0
                break

            # state = env.wrapped_env().gym.get_pybullet_state()
            time_step = env.step(action)
            # images.append(env.render(mode='rgb_array'))

        if time_step.reward < 1.0:
            print("Episode failed")
            num_failed_episodes += 1
            continue

        i += 1
        obs_history = np.array(obs_history)
        action_history = np.array(action_history)

        episode = {
            'obs': obs_history,
            'action': action_history
        }
        buffer.add_episode(episode)

        # # Save video
        # if not os.path.exists(output):
        #     os.makedirs(output)
        # imageio.mimsave(f'{output}/episode_{i}.gif', images)
        print("Length of episode:", len(obs_history), "Last reward:", time_step.reward)
    
    buffer.save_to_path(output)
    print("Number of failed episodes:", num_failed_episodes)
        
if __name__ == '__main__':
    main()
