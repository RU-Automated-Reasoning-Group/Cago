from pyrsistent import b
from gc_main import Method
import common
import torch
import pathlib
import tensorflow as tf
import gc_agent
from gc_goal_picker import *
import logging
import os
import re
import sys
import warnings
import imageio
from tqdm import tqdm
import math
import random
from functools import partial
import collections

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ruamel.yaml as yaml


class Workbench:


    def __init__(self, if_auto_load = True, if_load_agent = True):

        if if_auto_load:
            self._load_method(if_load_agent)

    def _load_method(self, if_load_agent = True):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger().setLevel('ERROR')
        warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

        sys.path.append(str(pathlib.Path(__file__).parent))
        sys.path.append(str(pathlib.Path(__file__).parent.parent))

        my_Method = Method()

        config = my_Method.Set_Config()

        
        env = my_Method.make_env(config)
        eval_env = my_Method.make_env(config, if_eval=True)

        obs2goal_fn = my_Method.make_obs2goal_fn(config)

        
        tf.config.run_functions_eagerly(not config.jit)
        # tf.config.run_functions_eagerly(True)  
        # tf.data.experimental.enable_debug_mode(not config.jit)
        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message

        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')  

        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        if config.precision == 16:
            from tensorflow.keras.mixed_precision import experimental as prec
            prec.set_policy(prec.Policy('mixed_float16'))


        
        logdir = pathlib.Path(config.logdir).expanduser()

        if if_load_agent:

            
            replay = common.Replay(logdir / 'train_episodes', **config.replay)  # initialize replay buffer

            
            # initialize step counter
            step = common.Counter(replay.stats['total_steps'])

            print('Create agent.')

            
            agnt = gc_agent.GCAgent(config, env.obs_space, env.act_space, step, obs2goal_fn)

            
            dataset = iter(replay.dataset(**config.dataset))  

            
            train_agent = common.CarryOverState(agnt.train)

            train_agent(next(dataset))

            
            if (logdir / 'variables.pkl').exists() and (logdir / 'cluster.pth').exists():
                print('Found existing checkpoint.')
                agnt.agent_load(logdir)

            self.replay = replay
            self.agnt = agnt

        self.my_Method = my_Method
        self.env = env
        self.eval_env = eval_env
        self.config = config
        self.logdir = logdir

        return 


    def load_method_from_logdir(self, logdir, if_load_agent = True):

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.getLogger().setLevel('ERROR')
        warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

        sys.path.append(str(pathlib.Path(__file__).parent))
        sys.path.append(str(pathlib.Path(__file__).parent.parent))

        my_Method = Method()

        logdir = pathlib.Path(logdir).expanduser()
        print(logdir)
        if logdir.exists():
            print('Loading existing config')
            yaml_config = yaml.safe_load((logdir / 'config.yaml').read_text())
            config = common.Config(yaml_config)

        else:
            raise ValueError("No such directory")

        
        env = my_Method.make_env(config)
        eval_env = my_Method.make_env(config, if_eval=True)

        obs2goal_fn = my_Method.make_obs2goal_fn(config)

        
        tf.config.run_functions_eagerly(False)
        # tf.config.run_functions_eagerly(True)  
        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message

        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        if config.precision == 16:
            from tensorflow.keras.mixed_precision import experimental as prec
            prec.set_policy(prec.Policy('mixed_float16'))

        if if_load_agent:

            
            replay = common.Replay(logdir / 'train_episodes', **config.replay)  # initialize replay buffer

            
            # initialize step counter
            step = common.Counter(replay.stats['total_steps'])

            print('Create agent.')

            
            agnt = gc_agent.GCAgent(config, env.obs_space, env.act_space, step, obs2goal_fn)

            
            dataset = iter(replay.dataset(**config.dataset))  

            
            train_agent = common.CarryOverState(agnt.train)

            train_agent(next(dataset))

            
            if (logdir / 'variables.pkl').exists():
                print('Found existing checkpoint.')
                agnt.agent_load(logdir)

            self.agnt = agnt
            self.replay = replay
        
        self.my_Method = my_Method
        self.config = config
        self.logdir = logdir
        self.env = env
        self.eval_env = eval_env

        return 


    def load_dataset(self):

        train_dataset = common.load_episodes(self.logdir / 'train_episodes', capacity=self.config.replay.capacity, minlen=self.config.replay.minlen)

        # eval_dataset = common.load_episodes(self.logdir / 'eval_episodes', capacity=self.config.replay.capacity, minlen=self.config.replay.minlen)

        return train_dataset, None
    

    def draw_demo_goal_distribution(self):

        result_img_path = self.logdir / 'demo_goal_distribution.png'

        self.train_dataset, self.eval_dataset = self.load_dataset()

        # Assuming self.train_dataset and self.eval_dataset are already loaded
        ep_demogoal_rate_list = []

        sample_rate = 1
        print(f"Sampling rate: {sample_rate}")

        None_count = 0

        selected_seed = None
        # Iterate through the training dataset
        for key, value in tqdm(self.train_dataset.items(), desc="Processing demogoal rate", unit="episode"):
            goal = value['goal'].tolist()[-1]
            demogoal_rate, seed = self.env.get_demogoal_index_rate(goal)

            if selected_seed is None:

                selected_seed = seed
                print(f"Selected seed: {selected_seed}")

            # print(f"Goal: {goal}")
            # print(f"Demogoal_rate: {demogoal_rate}")
            if demogoal_rate is not None:
                if random.random() < sample_rate:
                    ep_demogoal_rate_list.append(demogoal_rate)
                else:
                    continue
            else:
                None_count += 1
        
        print(f"None demogoal rate count: {None_count}")

        # Prepare data for plotting
        episode_numbers = range(1, len(ep_demogoal_rate_list) + 1)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(episode_numbers, ep_demogoal_rate_list, color='red', s=1)  # s controls the size of the points
        plt.title('Demo Goal Distribution')
        plt.xlabel('Episode Number')
        plt.ylabel('Demo Goal Rate')
        plt.grid(True)
        plt.savefig(result_img_path)
        plt.close()

        print(f"Demo goal distribution plot saved at: {result_img_path}")


    def main(self,):

        self.draw_demo_goal_distribution()

if __name__ == '__main__':

    My_Workbench = Workbench(if_auto_load = True, if_load_agent = False)
    My_Workbench.main()



