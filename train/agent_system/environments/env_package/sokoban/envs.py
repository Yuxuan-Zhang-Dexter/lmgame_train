import ray
import gym
# from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv
from agent_system.environments.env_package.sokoban.custom_02_sokoban.sokobanEnv import SokobanEnv
import numpy as np

@ray.remote(num_cpus=0.25)
class SokobanWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of SokobanEnv.
    """
    
    def __init__(self, mode, env_kwargs):
        """Initialize the Sokoban environment in this worker"""
        self.env = SokobanEnv(render_mode=mode, **env_kwargs)
        # Ensure the environment is properly initialized by calling reset
        self.env.reset()
    
    def step(self, action):
        """Execute a step in the environment"""
        # Convert action index to string action for the new SokobanEnv
        # The new env expects string actions like "up", "down", "push up", etc.
        # We'll map action indices to string actions
        action_mapping = {
            0: "no_op",
            1: "push up", 
            2: "push down",
            3: "push left", 
            4: "push right",
            5: "up",
            6: "down", 
            7: "left",
            8: "right"
        }
        
        action_str = action_mapping.get(action, "no_op")
        obs, reward, terminated, truncated, info, perf_score = self.env.step(action_str)
        return obs, reward, terminated or truncated, info
    
    def reset(self, seed_for_reset):
        """Reset the environment with given seed"""
        obs, info = self.env.reset(seed=seed_for_reset)
        return obs, info
    
    def render(self):
        """Render the environment"""
        # The new SokobanEnv render method doesn't take a mode parameter
        # It uses the render_mode set during initialization
        rendered = self.env.render()
        return rendered

@ray.remote(num_cpus=0.25)
class OldSokobanWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of SokobanEnv.
    """
    
    def __init__(self, mode, env_kwargs):
        """Initialize the Sokoban environment in this worker"""
        self.env = SokobanEnv(mode, **env_kwargs)
        self.mode = mode
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def reset(self, seed_for_reset):
        """Reset the environment with given seed"""
        obs, info = self.env.reset(seed=seed_for_reset)
        return obs, info
    
    def render(self):
        """Render the environment"""
        rendered = self.env.render(mode=self.mode)
        return rendered


class SokobanMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the Sokoban environment.
    Each Ray actor creates an independent SokobanEnv instance.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self,
                 seed=0, 
                 env_num=1, 
                 group_n=1, 
                 mode='rgb_array',
                 is_train=True,
                 env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing SokobanEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.mode = mode
        np.random.seed(seed)

        if env_kwargs is None:
            env_kwargs = {}

        # Create Ray remote actors instead of processes
        self.workers = []
        for i in range(self.num_processes):
            worker = SokobanWorker.remote(self.mode, env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[int], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        # randomly generate self.env_num seeds
        if self.is_train:
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # repeat the seeds for each group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode='rgb_array', env_idx=None):
        """
        Request rendering from Ray actor environments.
        Can specify env_idx to get render result from a specific environment,
        otherwise returns a list from all environments.
        """
        if env_idx is not None:
            future = self.workers[env_idx].render.remote()
            return ray.get(future)
        else:
            futures = []
            for worker in self.workers:
                future = worker.render.remote()
                futures.append(future)
            results = ray.get(futures)
            return results

    def close(self):
        """
        Close all Ray actors
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_sokoban_envs(
        seed=0,
        env_num=1,
        group_n=1,
        mode='rgb_array',
        is_train=True,
        env_kwargs=None):
    return SokobanMultiProcessEnv(seed, env_num, group_n, mode, is_train, env_kwargs=env_kwargs)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    env_kwargs={
        'dim_room': (6, 6),
        'num_boxes': 1,
        'max_steps_episode': 100,
        'game_specific_config_path_for_adapter': "agent_system/environments/env_package/sokoban/custom_02_sokoban/game_env_config_train.json", 
        'benchmark_mode': False,
        'tile_size_for_render': 4,
        'level_to_load': None,
    }
    env = SokobanWorker.remote('rgb_array', env_kwargs=env_kwargs)

    print("Resetting environment...")
    obs, info = ray.get(env.reset.remote(0))
    print("Environment reset. Initial observation and info:", info)

    seed = 0
    while True:
        np_img = ray.get(env.render.remote())
        plt.imsave('sokoban1.png', np_img)
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard.strip())
        result = ray.get(env.step.remote(action))
        obs, reward, done, info = result
        print(reward, done, info)
        if done:
            print("Resetting environment...")
            seed += 1
            obs, info = ray.get(env.reset.remote(seed))
            print("Environment reset. Initial observation and info:", obs, info)    
        
