

import atexit
from collections import defaultdict
import os
import sys
import threading
import traceback
import uuid
import io
import datetime
import pathlib

import cloudpickle
import gym
import numpy as np
from PIL import Image
import h5py
import pickle
import imageio
import json
from dreamerv2_demo.Goal_Predictor import Goal_Predictor, get_demo_trajectories, Find_important_dim
import random
import tensorflow as tf
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from mani_skill2.utils.sapien_utils import vectorize_pose

except:
    pass

class NormalizeActions:

    def __init__(self, env):
        self._env = env

        self._mask = np.logical_and(
                np.isfinite(env.action_space.low),
                np.isfinite(env.action_space.high))
        
        self._low = np.where(self._mask, env.action_space.low, -1)

        self._high = np.where(self._mask, env.action_space.high, 1)


    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)

        return self._env.step(original)


class NormObsWrapper:
    # 1. assumes we have observation, achieved_goal, desired_goal with same dims.
    # 2. we don't guarantee that normed obs is between [0, 1], since obs_min / obs_max are arbitrary bounds.
    def __init__(self, env, obs_min, obs_max, keys=None):
        self._env = env
        self.obs_min = obs_min
        self.obs_max = obs_max
        self.keys = keys

    def __getattr__(self, name):
        return getattr(self._env, name)

    def norm_ob_dict(self, ob_dict):
        ob_dict = ob_dict.copy()
        if self.keys is None:
            for k, v in ob_dict.items():
                ob_dict[k] = (v - self.obs_min) / (self.obs_max - self.obs_min)
        else:
            for k in self.keys:
                v = ob_dict[k]
                ob_dict[k] = (v - self.obs_min) / (self.obs_max - self.obs_min)
        return ob_dict


    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        return self.norm_ob_dict(obs), rew, done, info

    def reset(self):
        return self.norm_ob_dict(self._env.reset())

    def norm_ob(self, ob):
        return (ob - self.obs_min) / (self.obs_max - self.obs_min)

    def get_goals(self):
        goals = self._env.get_goals()
        norm_goals = np.stack([self.norm_ob(g) for g in goals])
        return norm_goals


class ConvertGoalEnvWrapper:
    """
    Given a GoalEnv that returns obs dict {'observation', 'achieved_goal', 'desired_goal'}, we modify obs dict to just contain {'observation', 'goal'} where 'goal' is desired goal.
    """
    def __init__(self, env, obs_key='observation', goal_key='goal'):
        self._env = env
        self.obs_key = obs_key
        self.goal_key = goal_key

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        assert self._obs_is_dict, "GoalEnv should have obs dict"

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        # toss out achieved_goal
        obs, reward, done, info = self._env.step(action)
        obs = {self.obs_key: obs[self.obs_key], self.goal_key: obs['desired_goal']}
        return obs, reward, done, info

    def reset(self):
        # toss out achieved_goal and desired_goal keys.
        obs = self._env.reset()
        obs = {self.obs_key: obs[self.obs_key], self.goal_key: obs['desired_goal']}
        return obs

    @property
    def observation_space(self):
        # just return dict with observation.
        return gym.spaces.Dict({self.obs_key: self._env.observation_space[self.obs_key], self.goal_key: self._env.observation_space["desired_goal"]})


class GymWrapper:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """
    def __init__(self, env, obs_key='image', act_key='action', info_to_obs_fn=None):
        self._env = env

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self.info_to_obs_fn = info_to_obs_fn

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        return {
                **spaces,
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }


    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, done, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = done
        obs['is_terminal'] = info.get('is_terminal', done)
        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(info, obs)
        return obs

    def reset(self):
        obs = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(None, obs)
        return obs
                     

# reset with assigned seed and assigned demo goal. goal space = obs space
class GymnasiumWrapper_Demo:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """
    def __init__(self, env, if_eval = False, reset_with_seed = False, obs_key='observation', act_key='action', info_to_obs_fn=None, config=None):

        self.if_use_obs2goal = False

        self._env = env

        # enlarge the hole size
        if self._env.spec.id == "PegInsertionSide-v0":

                self._env.env.env._clearance = 0.01

        self.if_eval = if_eval
        # self._env.obs_space['goal'] = env.obs_space['image']


        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self.info_to_obs_fn = info_to_obs_fn

        self.if_modify_obs_dim = config.if_modify_obs_dim
        self.goal_rate = 1
        self.goal_idx = 0

        self.config = config
        self.all_demo_trajectories, self.seed_list = get_demo_trajectories(config.task, config.demo_path, if_eval=False, if_random_seeds=True, random_seeds_num=150)
        assert len(self.all_demo_trajectories) == len(self.seed_list), "Demo trajectories and seed list should have the same length!"
        # print(len(self.all_demo_trajectories), len(self.seed_list))
        print("Used demo seed list:", self.seed_list, "Length:", len(self.seed_list))

        self.train_seed_num = self.config.train_seed_num
        self.train_seed_list = self.seed_list[:self.train_seed_num]

        self.if_count_demo_obs = True
        self.ep_observation = []
        self.ep_imgs = []

        self.train_seed_num = min(self.train_seed_num, len(self.train_seed_list))
        print("Train seed num:", self.train_seed_num)

        if self.if_count_demo_obs:
            self.demo_obs_count_dict = defaultdict(list)


            for traj_index in range(self.train_seed_num):
                trajectory = self.all_demo_trajectories[traj_index]
                num_obs = len(trajectory[self._obs_key]) 
                self.demo_obs_count_dict[traj_index] = [0.0] * num_obs

            # self.demo_obs_count_dict = 


        if "Adroit" in self._env.spec.id:

            self.init_state_dict_keys_dict = {
            "AdroitHandDoor-v1": ['qvel', 'qpos', 'door_body_pos'],
            "AdroitHandHammer-v1": ['qvel', 'qpos', 'board_pos'],
            "AdroitHandPen-v1": ['qvel', 'qpos', 'desired_orien'],
            "AdroitHandRelocate-v1": ['qvel', 'qpos', 'obj_pos', 'target_pos']
        }

        
        self.origanl_all_demo_trajectories = self.all_demo_trajectories.copy()


        if self.if_modify_obs_dim:

            for demo_tra in self.all_demo_trajectories:
                self.last_observation = None
                demo_tra['observation'] = np.apply_along_axis(self.modify_obs, axis=1, arr=demo_tra['observation'])

        if self.config.count_mode == "l2-important":
            self.important_obs_dim = Find_important_dim(self.all_demo_trajectories, env, find_way=3)
            self.important_obs_dim = self.important_obs_dim[:config.important_dim_num]
            print("Env use important obs dim:", self.important_obs_dim)

        # self.threshold_dict = self.calculate_threshold() 
        # print("Max threshold per trajectory:", {k: max(v) for k, v in self.threshold_dict.items()})

        self.reset_with_seed = reset_with_seed
        self.goal_idx = 0
        self.reset_seed = self.seed_list[self.goal_idx]
        self.goal, self.image_goal = self.get_goal()

        self.if_eval_rate_goal = False

        self.if_eval_random_seed_with_gp = False
        self.if_load_goal_predictor = True



        # Load goal predictor model

        try:
            gp_model_path = f'GP_MLP_Model/mlp_model_{self.config.task}.pth'

            if self.if_load_goal_predictor and gp_model_path is not None:
                
                self.goal_predictor = Goal_Predictor()
                if self.env.spec.id == "FetchPickAndPlace-v2":
                    obs_size = self.all_demo_trajectories[0]['observation'][0].shape[0] + 3
                else:
                    obs_size = self.all_demo_trajectories[0]['observation'][0].shape[0]
                self.goal_predictor.load_model(gp_model_path, obs_size, hidden_size=100)
                self.goal_predictor.gp_model.eval()

                print("Goal predictor model loaded successfully!")

        except Exception as e:

            self.goal_predictor = None
            print("Failed to load goal predictor model!, error:", e)

        self.agnt = None
    
        # save as npz
        self.demo_replaybuffer = True
        if self.demo_replaybuffer:

            demo_replaybuffer_dataset, seed_list = get_demo_trajectories(config.task, config.demo_path, if_eval=False, if_random_seeds=True, random_seeds_num=50)
            # demo_replaybuffer_dataset = demo_replaybuffer_dataset[:self.train_seed_num]

            if self.if_modify_obs_dim:

                for demo_tra in demo_replaybuffer_dataset:
                    self.last_observation = None
                    demo_tra['observation'] = np.apply_along_axis(self.modify_obs, axis=1, arr=demo_tra['observation'])
                    
            logdir = pathlib.Path(config.logdir).expanduser()
            npz_save_dir = logdir / 'demo_episodes'

            if not npz_save_dir.exists() and if_eval == False:

                npz_save_dir.mkdir(parents=True, exist_ok=True)

                for i, demo_ep in enumerate(demo_replaybuffer_dataset):
                    
                    if 'init_state_dict' in demo_ep:
                        del demo_ep['init_state_dict']
                    save_episode(npz_save_dir, demo_ep)
            
            else:

                print("Demo episodes npz directory already exist!")

        # for i, demo_tra in enumerate(self.all_demo_trajectories):

        #     self.reset()

            # for action in demo_tra['action']:
     
            #         # action = env.action_space.sample()  # Random policy
            #         # action = {'action': action}
                    
            #         obs, reward, terminated, truncated, info = self._env.step(action)

            #         img = self.render()

            #         imageio.imsave('demo_step.png', img)

            # for obs in demo_tra['observation']:

            #     img = self.render_with_obs(obs, None, width=200, height=200)

            #     imageio.imsave('demo_step_2.png', img)


    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)


    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {'observation': self._env.observation_space}

        if self.if_modify_obs_dim:
            
            spaces['observation'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.all_demo_trajectories[0]['observation'][0]),), dtype=np.float32)


        obs_space = {
                **spaces,
                'goal': spaces['observation'] if not self.if_use_obs2goal else gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        if self.config.if_image_obs:

            obs_space['image_obs'] = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            obs_space['image_goal'] = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        try:
            del obs_space['achieved_goal']
            del obs_space['desired_goal']
        except:
            pass

        return obs_space
    

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx if not self.if_eval_random_seed_with_gp else 999999
        obs['goal'] = self.goal
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = truncated
        obs['is_terminal'] = info.get('success', info.get('is_success', 'terminated'))
        obs['is_rate_obs_success'] = False

        if self._env.spec.id == "PegInsertionSide-v0" or self._env.spec.id == "StackCube-v0" or self._env.spec.id == "PickCube-v0" or self._env.spec.id == "PullCubeTool-v1" or self._env.spec.id == "OpenCabinetDrawer-v1" or self._env.spec.id == "PushChair-v1":
            obs['env_states'] = self._env.unwrapped.get_state()
            obs['env_seed'] = self.reset_seed
        elif "Adroit" in self._env.spec.id:
            env_state = self._env.unwrapped.get_env_state()
            obs['env_states'] = {key: env_state[key] for key in self.init_state_dict_keys_dict[self._env.spec.id]}
            obs['env_seed'] = self.reset_seed

            if self._env.spec.id == "AdroitHandDoor-v1":
                obs['is_terminal'] = True if obs[self._obs_key][-1] == 1 else False


        elif "meta" in self.config.task:
            obs['env_states'] = self._env.unwrapped.get_env_state()
            obs['env_seed'] = self.reset_seed
        else:
            obs['env_seed'] = self.reset_seed   
        
        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(info, obs)

        if self.if_modify_obs_dim:
            original_observation = obs[self._obs_key]
            obs[self._obs_key] = self.modify_obs(obs[self._obs_key]) 
        else:
            original_observation = obs[self._obs_key]

        if self.if_eval_rate_goal and not self.if_eval_random_seed_with_gp and not self.if_use_obs2goal:

            obs['is_rate_obs_success'] = self.is_rate_obs_success(obs) 
        
        # if self.if_count_demo_obs and self.goal_idx < self.train_seed_num and not self.if_eval:

        #     self.count_demo_obs(original_observation)

        if (self.if_count_demo_obs and self.goal_idx < self.train_seed_num and not self.if_eval) or self.config.if_image_obs:

            self.ep_observation.append(obs[self._obs_key])

            if self.config.count_mode == "img_distance" or self.config.if_image_obs:

                img = self._env.render()
                compressed_image = np.array(Image.fromarray(img.astype('uint8')).resize((100, 100)))
                self.ep_imgs.append(compressed_image)

                if self.config.if_image_obs:
                    obs['image_obs'] = compressed_image
                    obs['image_goal'] = self.image_goal

        return obs
    

    def count_demo_obs(self):

        count_mode = self.config.count_mode # "l2", "dd", "l2-vote", "latent_distance"

        time1 = time.time()

        if count_mode == "dd" and self.agnt is None:

            print("Dynamical distance mode is selected, but wm is None, use L2 count mode!")

            count_mode = "l2"

        if count_mode == "l2":

            # observation = obs[self._obs_key]
            demo_obs_seq = self.demo_tra_using[self._obs_key]
            ep_observation = np.array(self.ep_observation)

            if self._env.spec.id == "PegInsertionSide-v0":

                # observation_sub = np.concatenate((observation[7:9], observation[-25:]))
                # demo_obs_seq_sub = np.concatenate((demo_obs_seq[:, 7:9], demo_obs_seq[:, -25:]), axis=1)

                ep_observation_sub = ep_observation[:, -24:]
                demo_obs_seq_sub = demo_obs_seq[:, -24:]

                threshold = 0.05

            elif self._env.spec.id == "StackCube-v0" or self._env.spec.id == "PickCube-v0":

                ep_observation_sub = ep_observation[:, -30:]
                demo_obs_seq_sub = demo_obs_seq[:, -30:]

                threshold = 0.1

            elif self._env.spec.id == "OpenCabinetDrawer-v1":

                ep_observation_sub = ep_observation[:, -20:]
                demo_obs_seq_sub = demo_obs_seq[:, -20:]
                threshold = 0.05

            elif self._env.spec.id == "PushChair-v1":

                ep_observation_sub = np.concatenate((ep_observation[:, 39:45], ep_observation[:, -24:-12]), axis=1)
                demo_obs_seq_sub = np.concatenate((demo_obs_seq[:, 39:45], demo_obs_seq[:, -24:-12]), axis=1)

                threshold = 0.1

            elif self._env.spec.id == "AdroitHandDoor-v1":

                ep_observation_sub = ep_observation[:, 29:]
                demo_obs_seq_sub = demo_obs_seq[:, 29:]
                threshold = 0.2

            elif self._env.spec.id == "AdroitHandHammer-v1":

                # ep_observation_sub = np.concatenate((ep_observation[:, 6:7], ep_observation[:, 33:39], ep_observation[:, 42:]), axis=1)
                # demo_obs_seq_sub = np.concatenate((demo_obs_seq[:, 6:7], demo_obs_seq[:, 33:39], demo_obs_seq[:, 42:]), axis=1)

                ep_observation_sub = ep_observation[:, 26:]
                demo_obs_seq_sub = demo_obs_seq[:, 26:]

                threshold = 0.1

            elif self._env.spec.id == "AdroitHandPen-v1":

                # ep_observation_sub = np.concatenate((ep_observation[:, 24:27], ep_observation[:, 33:]), axis=1)
                # demo_obs_seq_sub = np.concatenate((demo_obs_seq[:, 24:27], demo_obs_seq[:, 33:]), axis=1)

                ep_observation_sub = ep_observation[:, 33:]
                demo_obs_seq_sub = demo_obs_seq[:, 33:]
                threshold = 0.2

            elif self._env.spec.id == "AdroitHandRelocate-v1":

                # ep_observation_sub = np.concatenate((ep_observation[:, 0:3], ep_observation[:, 30:]), axis=1)
                # demo_obs_seq_sub = np.concatenate((demo_obs_seq[:, 0:3], demo_obs_seq[:, 30:]), axis=1)
                ep_observation_sub = ep_observation[:, 30:]
                demo_obs_seq_sub = demo_obs_seq[:, 30:]
                threshold = 0.1  # 0.2 is too big for Relocate

            else:

                ep_observation_sub = ep_observation
                demo_obs_seq_sub = demo_obs_seq
                threshold = self.config.l2_threshold

            for obs_sub in ep_observation_sub:

                L2_distance = np.linalg.norm(demo_obs_seq_sub - obs_sub, axis=1)                
                valid_indices = np.where(L2_distance < threshold)[0]

                for idx in valid_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "l2-important":

            demo_obs_seq = self.demo_tra_using[self._obs_key]
            ep_observation = np.array(self.ep_observation)

            ep_observation_sub = ep_observation[:, self.important_obs_dim]
            demo_obs_seq_sub = demo_obs_seq[:, self.important_obs_dim]

            threshold = self.config.l2_important_threshold

            for obs_sub in ep_observation_sub:

                L2_distance = np.linalg.norm(demo_obs_seq_sub - obs_sub, axis=1)                
                valid_indices = np.where(L2_distance < threshold)[0]

                for idx in valid_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "l2-vote":

            demo_obs_seq = self.demo_tra_using[self._obs_key]
            ep_observation = np.array(self.ep_observation)
            threshold = 0.1 
            num_trials = 10  
            required_votes = 2 

            obs_dim = demo_obs_seq.shape[1]

            for obs_sub in ep_observation:
                vote_counts = np.zeros(demo_obs_seq.shape[0], dtype=int) 
                for _ in range(num_trials):
                    selected_dims = np.random.choice(obs_dim, size=5, replace=False)
                    L2_distance = np.linalg.norm(demo_obs_seq[:, selected_dims] - obs_sub[selected_dims], axis=1)
                    vote_counts += (L2_distance < threshold) 

                valid_indices = np.where(vote_counts >= required_votes)[0] 

                for idx in valid_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "dd":

            target_key  = ['observation', 'action', 'reward', 'is_last', 'is_first']

            demo_tra = {key: self.demo_tra_using[key] for key in target_key}
            demo_tra = {key: tf.identity(value) for key, value in demo_tra.items()}

            for key, value in demo_tra.items():
                demo_tra[key] = value[None, :]

            ep_observation = tf.stack(self.ep_observation, axis=0)  # shape: (batch_size, D)
            ep_observation = tf.identity(ep_observation)

            cur_obs = tf.tile(tf.expand_dims(ep_observation, axis=1), [1, demo_tra[self.config.state_key].shape[1], 1])  # (batch_size, T, D)

            cur_obs_dict = {self.config.state_key: cur_obs} 

            @tf.function
            def get_dd(cur_obs_dict, demo_tra):
                embed = self.agnt.wm.encoder(demo_tra)  # shape: (1, T, D)
                demo_embed = tf.cast(tf.squeeze(embed, axis=0), tf.float32)  # (T, D)

                cur_obs_embed = self.agnt.wm.encoder(cur_obs_dict)  # shape: (batch_size, T, D)
                cur_obs_embed = tf.cast(cur_obs_embed, tf.float32) 

                demo_embed_tiled = tf.tile(demo_embed[None, :, :], [cur_obs_embed.shape[0], 1, 1])

                dd_pred = self.agnt._task_behavior.dynamical_distance(tf.concat([cur_obs_embed, demo_embed_tiled], axis=-1))

                return dd_pred

            dd_pred = get_dd(cur_obs_dict, demo_tra).numpy() * self.config.imag_horizon  # shape: (batch_size, num_segments, 2*D)

            threshold = 3

            for i, dd in enumerate(dd_pred):  # dd shape: (num_segments, 2*D)
                valid_indices = np.where(dd < threshold)[0]

                if len(valid_indices) > 50:
                    print(f"invalid indices for ep_observation[{i}]:", valid_indices, dd)
                    continue

                for idx in valid_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "latent_distance":

            target_key  = ['observation', 'action', 'reward', 'is_last', 'is_first']

            demo_tra = {key: self.demo_tra_using[key] for key in target_key}
            demo_tra = {key: tf.identity(value) for key, value in demo_tra.items()}

            for key, value in demo_tra.items():
                demo_tra[key] = value[None, :] 

            ep_observation = tf.stack(self.ep_observation, axis=0)  # shape: (batch_size, D)
            ep_observation = tf.identity(ep_observation)

            cur_obs = tf.tile(tf.expand_dims(ep_observation, axis=1), [1, demo_tra[self.config.state_key].shape[1], 1])  # (batch_size, T, D)

            cur_obs_dict = {self.config.state_key: cur_obs}

            @tf.function
            def get_embed_distance(cur_obs_dict, demo_tra):

                demo_embed = self.agnt.wm.encoder(demo_tra)  # shape: (1, T, D)
                demo_embed = tf.cast(tf.squeeze(demo_embed, axis=0), tf.float32)  # (T, D)


                cur_obs_embed = self.agnt.wm.encoder(cur_obs_dict)  # shape: (batch_size, T, D)
                cur_obs_embed = tf.cast(cur_obs_embed, tf.float32)  # (batch_size, T, D)

                embed_distance = tf.norm(cur_obs_embed - demo_embed[None, :, :], axis=-1)  # (batch_size, T)

                return embed_distance

            embed_distance = get_embed_distance(cur_obs_dict, demo_tra).numpy()  # shape: (batch_size, num_segments)

            threshold = 0.5

            for i, dist in enumerate(embed_distance):  # dist shape: (num_segments,)
                valid_indices = np.where(dist < threshold)[0]

                if len(valid_indices) > 50:
                    print(f"invalid indices for ep_observation[{i}]:", valid_indices, dist)
                    continue

                for idx in valid_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "img_l2_distance":

            ep_imgs = np.array(self.ep_imgs)

            demo_img_seq = self.demo_tra_using['images']

            threshold = 0.1

            for i, img in enumerate(ep_imgs):

                L2_distance = np.linalg.norm(demo_img_seq - img, axis=(1, 2))

                valid_indices = np.where(L2_distance < threshold)[0]

                for idx in valid_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "img_distance":

            ep_imgs = np.array(self.ep_imgs)
            demo_img_seq = np.array(self.demo_tra_using['images'])

            # threshold = 0.95 
            mse_threshold = self.config.img_distance_threshold

            from skimage.metrics import mean_squared_error as img_mse

            # def mse_filter(img, demo_imgs, demo_indices):
            #     filtered = [(demo_img, idx) for demo_img, idx in zip(demo_imgs, demo_indices) 
            #                 if img_mse(img, demo_img) < mse_threshold]
            #     return zip(*filtered) if filtered else ([], [])  
            
            def mse_filter(img, demo_imgs, demo_indices):
                mse = [(idx, img_mse(img, demo_img)) for demo_img, idx in zip(demo_imgs, demo_indices)]
                filtered = [(idx, mse) for idx, mse in mse if mse < mse_threshold]
                # filtered = [(idx, mse) for demo_img, idx in zip(demo_imgs, demo_indices) if (mse := img_mse(img, demo_img)) < mse_threshold]
                return zip(*filtered) if filtered else ([], [])

            for i, img in enumerate(ep_imgs):

                filtered_indices, filtered_mses = mse_filter(img, demo_img_seq, np.arange(len(demo_img_seq)))

                if len(filtered_indices) == 0:
                    continue 

                # img_ssim_scores = np.array([ssim(img, demo_img, win_size=7, channel_axis=-1) for demo_img in candidate_demo_imgs])

                # valid_indices = np.where(img_ssim_scores > threshold)[0]

                for idx in filtered_indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1

        elif count_mode == "img_distance_matrix":
        
            ep_imgs = np.array(self.ep_imgs)
            demo_img_seq = np.array(self.demo_tra_using['images'])

            mse_threshold = self.config.img_distance_threshold 

            mse_values = np.mean((ep_imgs[:, None, :, :, :] - demo_img_seq[None, :, :, :, :]) ** 2, axis=(2, 3, 4))

            valid_mask = mse_values < mse_threshold
            filtered_indices = [np.where(valid_mask[i])[0] for i in range(len(ep_imgs))]

            for i, indices in enumerate(filtered_indices):
                for idx in indices:
                    self.demo_obs_count_dict[self.goal_idx][idx] += 1


        time2 = time.time()

        print(f"Count mode: {count_mode}, time cost: {time2 - time1:.4f}s")


    def calculate_threshold(self):
    
        threshold_dict = {}

        for traj_index in range(self.train_seed_num):
            trajectory = self.all_demo_trajectories[traj_index]
            obs_values = trajectory[self._obs_key]
            num_obs = len(obs_values)
            
            threshold_dict[traj_index] = [0.0] * num_obs
            
            for obs_index in range(num_obs):
                obs_sub = obs_values[obs_index][self.important_obs_dim]

                start_idx = max(0, obs_index - 5)
                end_idx = min(num_obs, obs_index + 6)

                max_l2_dist = 0.0
                for neighbor_idx in range(start_idx, end_idx):
                    if neighbor_idx != obs_index: 
                        neighbor_obs_sub = obs_values[neighbor_idx][self.important_obs_dim]
                        l2_dist = np.linalg.norm(neighbor_obs_sub - obs_sub)
                        max_l2_dist = max(max_l2_dist, l2_dist)

                threshold_dict[traj_index][obs_index] = max_l2_dist
            
        return threshold_dict


    def is_rate_obs_success(self, obs):

        assert obs['observation'].shape == obs['goal'].shape

        if self._env.spec.id == "PegInsertionSide-v0" or self._env.spec.id == "StackCube-v0" or self._env.spec.id == "PickCube-v0" or self._env.spec.id == "PullCubeTool-v1" or self._env.spec.id == "OpenCabinetDrawer-v1" or self._env.spec.id == "PushChair-v1":

            abs_diff = np.abs(obs['goal'] - obs['observation'])

            if np.all(abs_diff < 0.05):
                return True
            else:
                return False
        
        elif self._env.spec.id == "FetchPickAndPlace-v2":

            abs_diff = np.abs(obs['goal'] - obs['observation'])

            if np.all(abs_diff < 0.05):
                return True
            else:
                return False
        
        elif "Adroit" in self._env.spec.id:

            abs_diff = np.abs(obs['goal'] - obs['observation'])

            if np.all(abs_diff < 0.05):
                return True
            else:
                return False
                        
            # L2_distance = np.linalg.norm(obs['goal'] - obs['observation'])

            # if L2_distance < 0.5:
            #     return True
            # else:
            #     return False

        elif "meta" in self.config.task:

            abs_diff = np.abs(obs['goal'] - obs['observation'])

            if np.all(abs_diff < 0.05):
                return True
            else:
                return False
        else:
            raise NotImplementedError


    @ property
    def init_state_dict(self):

        return self.all_demo_trajectories[self.goal_idx]["init_state_dict"]
    
    def reset(self):

        if self.if_eval_random_seed_with_gp:

            self.reset_seed = np.random.randint(1, 10001)

            obs, info = self._env.reset(seed=self.reset_seed)

        else:

            if self.reset_with_seed:
                if "Adroit" in self._env.spec.id:
                    obs, info = self._env.reset()
                    self._env.unwrapped.set_env_state(self.init_state_dict)
                else:
                    if self._env.spec.id == "OpenCabinetDrawer-v1" or self._env.spec.id == "PushChair-v1" or self._env.spec.id == "PullCubeTool-v1":
                        obs, info = self._env.reset(seed = self.reset_seed, options=dict(reconfigure=True))
                    else:
                        obs, info = self._env.reset(seed = self.reset_seed)

            else:
                goal_idx = np.random.choice(self.train_seed_num)
                self.set_goal_idx(goal_idx)
                if "Adroit" in self._env.spec.id:
                    obs, info = self._env.reset()
                    self._env.unwrapped.set_env_state(self.init_state_dict)
                else:            
                    if self._env.spec.id == "OpenCabinetDrawer-v1" or self._env.spec.id == "PushChair-v1" or self._env.spec.id == "PullCubeTool-v1":
                        obs, info = self._env.reset(seed = self.reset_seed, options=dict(reconfigure=True))
                    else:
                        obs, info = self._env.reset(seed = self.reset_seed)

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if self._env.spec.id == "PegInsertionSide-v0" or self._env.spec.id == "StackCube-v0" or self._env.spec.id == "PickCube-v0" or self._env.spec.id == "PullCubeTool-v1" or self._env.spec.id == "OpenCabinetDrawer-v1" or self._env.spec.id == "PushChair-v1":
            obs['env_states'] = self._env.unwrapped.get_state()
            obs['env_seed'] = self.reset_seed
        
        elif "Adroit" in self._env.spec.id:
            env_state = self._env.unwrapped.get_env_state()
            obs['env_states'] = {key: env_state[key] for key in self.init_state_dict_keys_dict[self._env.spec.id]}
            obs['env_seed'] = self.reset_seed
        
        elif "meta" in self.config.task:
            obs['env_states'] = self._env.unwrapped.get_env_state()
            obs['env_seed'] = self.reset_seed
        else:
            obs['env_seed'] = self.reset_seed   

        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(None, obs)

        if self.if_modify_obs_dim:
            original_observation = obs[self._obs_key]
            obs[self._obs_key] = self.modify_obs(obs[self._obs_key])

        else:
            original_observation = obs[self._obs_key]

        if self.if_eval_random_seed_with_gp:

            self.goal_idx = 999999

            if self._env.spec.id == "FetchPickAndPlace-v2":
                initial_obs = np.concatenate((obs[self._obs_key], obs['desired_goal']))
                self.goal = self.goal_predictor.predict_goal(initial_obs)[0]
                self.goal = self.goal[:-3]
            else:
                initial_obs = obs[self._obs_key]
                self.goal = self.goal_predictor.predict_goal(initial_obs)[0]

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = self.goal
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        obs['is_rate_obs_success'] = False

        self.ep_observation = [obs[self._obs_key]]

        if (self.if_count_demo_obs and self.goal_idx < self.train_seed_num and not self.if_eval) or self.config.if_image_obs:

            if self.config.count_mode == "img_distance" or self.config.if_image_obs:

                img = self._env.render()
                compressed_image = np.array(Image.fromarray(img.astype('uint8')).resize((100, 100)))

                if self.config.count_mode == "img_distance":
                    self.ep_imgs = [compressed_image]

                if self.config.if_image_obs:
                    obs['image_obs'] = compressed_image
                    obs['image_goal'] = self.image_goal


        self.last_observation = None
        
        return obs
    

    def modify_obs(self, obs):

        if self._env.spec.id == "PegInsertionSide-v0":

            if self.last_observation is None:

                self.last_observation = obs

            assert obs.shape[0] == 50
            # print(obs)
            # obs = np.concatenate([obs[:9], obs[25:]])
            # demo_tra['observation'] = np.concatenate((demo_tra['observation'][:, :9], demo_tra['observation'][:, 16:18], demo_tra['observation'][:, 25:]), axis=1) 
            # obs = np.concatenate((obs[:9], obs[16:18], obs[25:]))
            # obs = obs[25:]
            # print(obs)

            # tcp_pose = obs[25:32]
            # peg_pose = obs[32:39]
            # hole_pose = obs[42:49]

            # new_part_1 = tcp_pose - peg_pose
            # new_part_2 = peg_pose - hole_pose

            # obs_sub = np.concatenate((self.last_observation[7:9], self.last_observation[25:39], obs[7:9], obs[25:-1]))
            # obs_sub = np.concatenate((self.last_observation[:-11], obs[:-1]))
            # obs_sub = np.concatenate((obs[7:9], obs[-25:-1]))

            obs[7] = obs[7]/0.04  # A normalized measurement of how open the gripper is
            obs[8] = obs[8]/0.04  # A normalized measurement of how open the gripper is

            obs_sub = obs[:-1]

            obs_sub[18:21] *= 10
            obs_sub[25:28] *= 10
            obs_sub[32:35] *= 10
            obs_sub[42:45] *= 10

            self.last_observation = obs

            # obs = np.concatenate([obs, new_part_1, new_part_2], axis=0)

            return obs_sub
        
        elif self._env.spec.id == "StackCube-v0":

            assert obs.shape[0] == 55

            if self.last_observation is None:

                self.last_observation = obs
            
            # obs_sub = obs[-30:]
            # obs_sub = np.concatenate((self.last_observation[7:9], self.last_observation[-30:], obs[7:9], obs[-30:]))
            # obs_sub = np.concatenate((self.last_observation[:-11], obs[:-1]))
            # obs_sub = np.concatenate((obs[7:9], obs[-30:]))

            obs_sub = obs
            obs_sub[7] = obs_sub[7]/0.04
            obs_sub[8] = obs_sub[8]/0.04  # A normalized measurement of how open the gripper is
            obs_sub[18:21] *= 10
            obs_sub[25:28] *= 10
            obs_sub[32:35] *= 10
            obs_sub[39:42] *= 10
            obs_sub[46:] *= 10

            # obs_sub = np.concatenate((obs_sub[0:9], obs_sub[18:]))
            self.last_observation = obs

            return obs_sub

        elif self._env.spec.id == "PickCube-v0":

            assert obs.shape[0] == 51

            if self.last_observation is None:

                self.last_observation = obs
            
            # obs_sub = obs[-30:]
            # obs_sub = np.concatenate((self.last_observation[7:9], self.last_observation[-30:], obs[7:9], obs[-30:]))
            # obs_sub = np.concatenate((self.last_observation[:-11], obs[:-1]))
            # obs_sub = np.concatenate((obs[7:9], obs[-30:]))

            obs_sub = obs
            obs_sub[7] = obs_sub[7]/0.04
            obs_sub[8] = obs_sub[8]/0.04  # A normalized measurement of how open the gripper is
            obs_sub[18:21] *= 10
            obs_sub[25:28] *= 10
            obs_sub[32:41] *= 10
            obs_sub[45:51] *= 10

            # obs_sub = np.concatenate((obs_sub[7:9], obs_sub[25:]))

            self.last_observation = obs

            return obs_sub
        
        elif self._env.spec.id == "PullCubeTool-v1":

            assert obs.shape[0] == 39

            if self.last_observation is None:

                self.last_observation = obs

            # obs_sub = obs[-30:]
            # obs_sub = np.concatenate((self.last_observation[7:9], self.last_observation[-30:], obs[7:9], obs[-30:]))
            # obs_sub = np.concatenate((self.last_observation[:-11], obs[:-1]))
            # obs_sub = np.concatenate((obs[7:9], obs[-30:])) 

            obs_sub = obs
            obs_sub[7] = obs_sub[7]/0.04
            obs_sub[8] = obs_sub[8]/0.04  # A normalized measurement of how open the gripper is
            obs_sub[18:21] *= 10
            obs_sub[25:28] *= 10
            obs_sub[32:35] *= 10


            # obs_sub = np.concatenate((obs_sub[0:9], obs_sub[18:]))

            self.last_observation = obs

            return obs_sub
            
        else:
            
            return obs


    def set_goal_idx(self, goal_idx):

        self.goal_idx = goal_idx
        self.reset_seed = self.seed_list[self.goal_idx]
        self.goal, image_goal = self.get_goal()

        if self.config.if_image_obs:

            self.image_goal = image_goal


    def get_demogoal_render_state(self, goal_idx, goal):

        if self._env.spec.id == "PegInsertionSide-v0" or self._env.spec.id == "StackCube-v0" or self._env.spec.id == "PickCube-v0" or self._env.spec.id == "PullCubeTool-v1" or self._env.spec.id == "OpenCabinetDrawer-v1" or self._env.spec.id == "PushChair-v1" or "Adroit" in self._env.spec.id or "meta" in self.config.task:

            try:
                demo_tra = self.all_demo_trajectories[goal_idx]
                # for demo_tra_idx in range(len(self.all_demo_trajectories)):
                #     demo_tra = self.all_demo_trajectories[demo_tra_idx]
                observations = demo_tra['observation']
                states = demo_tra['env_states']
                
                # print("observation:", observations)
                # print("states:", states)
                
                atol = 1e-5 
                indices = np.where(np.all(np.isclose(observations, goal, atol=atol), axis=1))[0]
                
                if len(indices) > 0:
                    index = indices[0]

                    render_state = states[index]
                    # print('demo_tra_idx', demo_tra_idx)
                    # print('goal_idx', self.goal_idx)
                    # print('reset_seed', self.reset_seed)
                    # self.set_goal_idx(demo_tra_idx)
                    # print("render_state:", render_state)
                    return render_state  
                    # else:
                    #     continue

                return None
            
            except Exception as e:

                return None


    def get_demogoal_index_rate(self, goal):

        try:
            for demo_tra_idx in range(len(self.all_demo_trajectories)):
                demo_tra = self.all_demo_trajectories[demo_tra_idx]

                observations = demo_tra['observation']

                atol = 1e-8  

                indices = np.where(np.all(np.isclose(observations, goal, atol=atol), axis=1))[0]
                # print("indices:", indices)
                
                if len(indices) > 0:
                    index = indices[-1]

                    index_rate = index / len(observations)

                    return index_rate, self.seed_list[demo_tra_idx]
                else:
                    continue

            return None, None
        
        except Exception as e:

            return None, None


    def get_goal(self):

        demo_tra = self.all_demo_trajectories[self.goal_idx]
        self.demo_tra_using = demo_tra
        demo_obs_seq = demo_tra[self._obs_key]
        demo_obs_num = len(demo_obs_seq)
        goal_obs_idx = min(int(demo_obs_num * self.goal_rate), demo_obs_num - 1)
        goal = demo_obs_seq[goal_obs_idx]

        if self.if_use_obs2goal:

            goal = self.obs2goal(goal)

        if self.config.if_image_obs:

            image_goal = demo_tra['images'][goal_obs_idx]

            return goal, image_goal

        return goal, None


    # compress image
    def render(self, width=200, height=200):

        image = self._env.render()



        # if self.env.spec.id == "PullCubeTool-v1":

        #     image = image.cpu().numpy()
        #     image = np.squeeze(image)

        compressed_image = Image.fromarray(image.astype('uint8')).resize((width, height))
        # compressed_image = image

        return compressed_image


    def render_with_obs(self, obs, goal, width=200, height=200):

        if self._env.spec.id == "FetchPickAndPlace-v2" or self._env.spec.id == 'FetchPush-v2' or self._env.spec.id == "FetchSlide-v2":

            # image = self.render(width, height)  # render the environment

            # imageio.imsave('render.png', image)

            inner_env = self._env.env.env.env

            inner_env.reset()

            data = inner_env.data
            model = inner_env.model

            object_pos = obs[3:6]  # position of the block
            gripper_target = obs[:3]  # position of the gripper

            gripper_right_finger = obs[9]  # right finger of the gripper
            gripper_left_finger = obs[10]  # left finger of the gripper
        
            if goal is not None:
                
                inner_env.goal = goal  # set the goal for the environment

            inner_env._utils.set_mocap_pos(model, data, "robot0:mocap", gripper_target)  # set the position of the gripper

            for _ in range(10):
                inner_env._mujoco.mj_step(model, data, nstep=inner_env.n_substeps)  # make sure the gripper is in the correct position


            inner_env._utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", gripper_right_finger)  # set right finger
            inner_env._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", gripper_left_finger)  # set left finger

            object_qpos = inner_env._utils.get_joint_qpos(
                        model, data, "object0:joint"
                    )
            
            assert object_qpos.shape == (7,)

            object_qpos[:3] = object_pos  

            inner_env._utils.set_joint_qpos(model, data, "object0:joint", object_qpos)  # set the position of the block
            
            inner_env._mujoco.mj_forward(model, data)  # update the simulator

            image = self.render(width, height)  # render the environment

            # imageio.imsave('FetchPickAndPlace-v2.png', image)
        
        elif "HandManipulateBlock" in self._env.spec.id or "HandManipulatePen" in self._env.spec.id:

            if self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1':
        
                goal = goal[-4:]
                goal = np.concatenate((np.array([1, 0.87, 0.17]), goal))

            inner_env = self._env.env.env.env

            data = inner_env.data
            model = inner_env.model

            block_qpos =  obs[-7:]
            inner_env._utils.set_joint_qpos(model, data, "object:joint", block_qpos)


            hand_block_target_qpos = np.concatenate((obs[:24], block_qpos, goal))

            inner_env.goal = goal

            data.qpos[:] = np.copy(hand_block_target_qpos)

            if model.na != 0:
                data.act[:] = None

            inner_env._mujoco.mj_forward(model, data)

            image = self.render(width, height) 

            return image
        
        else:
            
            raise NotImplementedError

        return image


    def obs2goal(self, obs):
        
        if self._env.spec.id == "PegInsertionSide-v0":

            return obs[..., -18:-11]

        else:

            raise NotImplementedError


# reset with assigned seed. goal from obs['desired_goal']
class GymnasiumWrapper_0:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """
    def __init__(self, env, if_eval = False, reset_with_seed = False, obs_key='observation', act_key='action', info_to_obs_fn=None, if_modify_obs_dim=False):
        self._env = env

        self.if_eval = if_eval
        # self._env.obs_space['goal'] = env.obs_space['image']


        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')

        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key
        self.info_to_obs_fn = info_to_obs_fn

        self.if_modify_obs_dim = if_modify_obs_dim

        if self._env.spec.id == "FetchPickAndPlace-v2" or self._env.spec.id == "PegInsertionSide-v0" or self._env.spec.id == 'FetchPush-v2':

            self.goal_idx = 0

            if self.if_eval:

                self.seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            else:

                self.seed_list = [0, 1, 2, 3, 4,]

        self.reset_with_seed = reset_with_seed
        self.goal_idx = 0
        self.reset_seed = self.seed_list[self.goal_idx]


    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        if self.if_modify_obs_dim:
            spaces[self._obs_key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        obs_space = {
                **spaces,
                'goal': spaces['desired_goal'],
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        del obs_space['achieved_goal']
        del obs_space['desired_goal']

        return obs_space

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = obs['desired_goal']
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = truncated
        obs['is_terminal'] = info['is_success']

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['env_states'] = self._env.unwrapped.get_state()

        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(info, obs)

        if self.if_modify_obs_dim:
            obs[self._obs_key] = self.modify_obs(obs[self._obs_key]) 
        
        return obs

    def reset(self):
        if self.reset_with_seed:
            obs, info = self._env.reset(seed = self.reset_seed)
        else:
            # obs, info = self._env.reset()
            goal_idx = np.random.choice(len(self.seed_list))
            self.set_goal_idx(goal_idx)
            obs, info = self._env.reset(seed = self.reset_seed)

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        obs['goal_idx'] = self.goal_idx
        obs['goal'] = obs['desired_goal']
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['env_states'] = self._env.unwrapped.get_state()

        if self.info_to_obs_fn:
            obs = self.info_to_obs_fn(None, obs)

        if self.if_modify_obs_dim:
            obs[self._obs_key] = self.modify_obs(obs[self._obs_key]) 
        
        return obs
    

    def modify_obs(self, obs):

        if self._env.spec.id == "PegInsertionSide-v0":
            # print(obs)
            # obs = np.concatenate([obs[:9], obs[18:42]])
            obs = obs[25:42]
            # print(obs)

            return obs
        else:
            raise NotImplementedError


    def set_goal_idx(self, goal_idx):

        self.goal_idx = goal_idx
        self.reset_seed = self.seed_list[self.goal_idx]


    def render(self, width=200, height=200):

        image = self._env.render()

        compressed_image = Image.fromarray(image.astype('uint8')).resize((width, height))

        return compressed_image

# random reset
class GymnasiumWrapper_1:
    """modifies obs space, action space,
    modifies step and reset fn, just returns ob dict now. ob dict has reward, done, etc.
    """
    def __init__(self, env, if_eval = False, obs_key='observation', act_key='action'):
        self._env = env

        self.if_eval = if_eval
        # self._env.obs_space['goal'] = env.obs_space['image']

        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')

        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}

        obs_space = {
                **spaces,
                'goal': gym.spaces.Box(-np.inf, np.inf, (7, ), dtype=np.float32) if self._env.spec.id == "PegInsertionSide-v0" else spaces['desired_goal'],
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        if self._env.spec.id == "PegInsertionSide-v0":

            obs_space['goal'] = gym.spaces.Box(-np.inf, np.inf, (7, ), dtype=np.float32)

        elif self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1':
        
            obs_space['goal'] = gym.spaces.Box(-np.inf, np.inf, (4, ), dtype=np.float32)

        else:

            obs_space['goal'] = spaces['desired_goal']

        if self._env.spec.id != "PegInsertionSide-v0":

            del obs_space['achieved_goal']
            del obs_space['desired_goal']

        return obs_space
    

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

        def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        
        if self._env.spec.id == "PegInsertionSide-v0":

            obs['goal'] = self.goal
            obs['env_states'] = self._env.unwrapped.get_state()

        elif self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1':
        
            obs['goal'] = obs['desired_goal'][-4:]

        else:
            obs['goal'] = obs['desired_goal']
        
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = truncated

        if self._env.spec.id == "PegInsertionSide-v0":
            obs['is_terminal'] = info['success']
        else:
            obs['is_terminal'] = info['is_success']
        
        return obs

        def reset(self):

        obs, info = self._env.reset()

        if not self._obs_is_dict:
            obs = {self._obs_key: obs}

        if self._env.spec.id == "PegInsertionSide-v0":

            self.goal = vectorize_pose(self._env.box_hole_pose)
            obs['goal'] = self.goal
            obs['env_states'] = self._env.unwrapped.get_state()

        elif self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1':
        
            obs['goal'] = obs['desired_goal'][-4:]
        
        else:
            obs['goal'] = obs['desired_goal']
        
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False

        
        return obs


    def render(self, width=200, height=200):

        image = self._env.render()

        compressed_image = Image.fromarray(image.astype('uint8')).resize((width, height))

        return compressed_image


    def render_with_obs(self, obs, goal, width=200, height=200):

        if self._env.spec.id == "FetchPickAndPlace-v2" or self._env.spec.id == 'FetchPush-v2' or self._env.spec.id == "FetchSlide-v2":

            inner_env = self._env.env.env.env

            data = inner_env.data
            model = inner_env.model

            object_pos = obs[3:6]  # position of the block
            gripper_target = obs[:3]  # position of the gripper

            gripper_right_finger = obs[9]  # right finger of the gripper
            gripper_left_finger = obs[10]  # left finger of the gripper
        
            inner_env.goal = goal  # set the goal for the environment

            inner_env._utils.set_mocap_pos(model, data, "robot0:mocap", gripper_target)  # set the position of the gripper

            for _ in range(10):
                inner_env._mujoco.mj_step(model, data, nstep=inner_env.n_substeps)  # make sure the gripper is in the correct position


            inner_env._utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", gripper_right_finger)  # set right finger
            inner_env._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", gripper_left_finger)  # set left finger

            object_qpos = inner_env._utils.get_joint_qpos(
                        model, data, "object0:joint"
                    )
            
            assert object_qpos.shape == (7,)

            object_qpos[:3] = object_pos  

            inner_env._utils.set_joint_qpos(model, data, "object0:joint", object_qpos)  # set the position of the block
            
            inner_env._mujoco.mj_forward(model, data)  # update the simulator

            image = self.render(width, height)  # render the environment
        
        elif "HandManipulateBlock" in self._env.spec.id or "HandManipulatePen" in self._env.spec.id:

            if self._env.spec.id == 'HandManipulateBlockRotateXYZ-v1' or self._env.spec.id == 'HandManipulatePenRotate-v1':
        
                goal = goal[-4:]
                goal = np.concatenate((np.array([1, 0.87, 0.17]), goal))

            inner_env = self._env.env.env.env

            data = inner_env.data
            model = inner_env.model

            block_qpos =  obs[-7:]
            inner_env._utils.set_joint_qpos(model, data, "object:joint", block_qpos)


            hand_block_target_qpos = np.concatenate((obs[:24], block_qpos, goal))

            inner_env.goal = goal

            data.qpos[:] = np.copy(hand_block_target_qpos)

            if model.na != 0:
                data.act[:] = None

            inner_env._mujoco.mj_forward(model, data)

            image = self.render(width, height) 

            return image
        
        else:
            
            raise NotImplementedError

        return image


class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):

        
        os.environ['MUJOCO_GL'] = 'egl'


        domain, task = name.split('_', 1)  

        
        if domain == 'cup':    # Only domain with multiple words.
            domain = 'ball_in_cup'

        if domain == 'manip':
            from dm_control import manipulation
            self._env = manipulation.load(task + '_vision')

        elif domain == 'locom':
            from dm_control.locomotion.examples import basic_rodent_2020
            self._env = getattr(basic_rodent_2020, task)()

        else:
            from dm_control import suite
            self._env = suite.load(domain, task)

        
        self._action_repeat = action_repeat

        
        self._size = size


        
        if camera in (-1, None):
            camera = dict(
                    quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
                    quadruped_fetch=2, locom_rodent_maze_forage=1,
                    locom_rodent_two_touch=1,
            ).get(name, 0)
        self._camera = camera


        
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

    
    @property
    def obs_space(self):

        
        spaces = {
                'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),  # (64, 64) + (3,) = (64, 64, 3)
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

        for key, value in self._env.observation_spec().items():

            if key in self._ignored_keys:
                continue

            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)

            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)

            else:
                raise NotImplementedError(value.dtype)
            
        return spaces

    
    @property
    def act_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return {'action': action}

    
    def step(self, action):

        
        
        assert np.isfinite(action['action']).all(), action['action']


        reward = 0.0  

        for _ in range(self._action_repeat):
            time_step = self._env.step(action['action'])
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        assert time_step.discount in (0, 1)

        
        obs = {
                'reward': reward,
                'is_first': False,
                'is_last': time_step.last(),
                'is_terminal': time_step.discount == 0,
                'image': self._env.physics.render(*self._size, camera_id=self._camera),
        }

        
        obs.update({
                k: v for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys})
        
        
        return obs

    
    def reset(self):
        time_step = self._env.reset()
        obs = {
                'reward': 0.0,
                'is_first': True,
                'is_last': False,
                'is_terminal': False,
                'image': self._env.physics.render(*self._size, camera_id=self._camera),
        }
        obs.update({
                k: v for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys})
        return obs



class Atari:

    LOCK = threading.Lock()  

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky=True, all_actions=False):
        
        assert size[0] == size[1]  

        import gym.wrappers
        import gym.envs.atari


        if name == 'james_bond':
            name = 'jamesbond'
        
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                    game=name, obs_type='image', frameskip=1,
                    repeat_action_probability=0.25 if sticky else 0.0,
                    full_action_space=all_actions)  
            
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None  

        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        
        
        self._env = gym.wrappers.AtariPreprocessing(
                env, noops, action_repeat, size[0], life_done, grayscale)  
        
        self._size = size
        self._grayscale = grayscale

    
    @property
    def obs_space(self):
        shape = self._size + (1 if self._grayscale else 3,) 
        return {
                'image': gym.spaces.Box(0, 255, shape, np.uint8),
                'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    
    @property
    def act_space(self):
        return {'action': self._env.action_space}

    
    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        if self._grayscale:
            image = image[..., None]
        return {
                'image': image,
                'ram': self._env.env._get_ram(),
                'reward': reward,
                'is_first': False,
                'is_last': done,
                'is_terminal': done,
        }

    
    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]  
        return {
                'image': image,
                'ram': self._env.env._get_ram(),
                'reward': 0.0,
                'is_first': True,
                'is_last': False,
                'is_terminal': False,
        }

    
    def close(self):
        return self._env.close()



class Crafter:

    def __init__(self, outdir=None, reward=True, seed=None):
        import crafter
        self._env = crafter.Env(reward=reward, seed=seed)
        self._env = crafter.Recorder(
                self._env, outdir,
                save_stats=True,
                save_video=False,
                save_episode=False,
        )
        self._achievements = crafter.constants.achievements.copy()

    @property
    def obs_space(self):
        spaces = {
                'image': self._env.observation_space,
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
        }
        spaces.update({
                f'log_achievement_{k}': gym.spaces.Box(0, 2 ** 31 - 1, (), np.int32)
                for k in self._achievements})
        return spaces

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        obs = {
                'image': image,
                'reward': reward,
                'is_first': False,
                'is_last': done,
                'is_terminal': info['discount'] == 0,
                'log_reward': info['reward'],
        }
        obs.update({
                f'log_achievement_{k}': v
                for k, v in info['achievements'].items()})
        return obs

    def reset(self):
        obs = {
                'image': self._env.reset(),
                'reward': 0.0,
                'is_first': True,
                'is_last': False,
                'is_terminal': False,
                'log_reward': 0.0,
        }
        obs.update({
                f'log_achievement_{k}': 0
                for k in self._achievements})
        return obs




class Dummy:

    def __init__(self):
        pass

    @property
    def obs_space(self):
        return {
                'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
                'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
                'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
                'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        }

    @property
    def act_space(self):
        return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

    def step(self, action):
        return {
                'image': np.zeros((64, 64, 3)),
                'reward': 0.0,
                'is_first': False,
                'is_last': False,
                'is_terminal': False,
        }

    def reset(self):
        return {
                'image': np.zeros((64, 64, 3)),
                'reward': 0.0,
                'is_first': True,
                'is_last': False,
                'is_terminal': False,
        }



class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs['is_last'] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()



class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])

        return self._env.step({**action, self._key: orig})
        


class OneHotAction:

    def __init__(self, env, key='action'):
        assert hasattr(env.act_space[key], 'n')
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    
    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    
    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    
    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference



class ResizeImage:

    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size

        
        
        self._keys = [
                k for k, v in env.obs_space.items()
                if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    
    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    
    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    
    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    
    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image



class RenderImage:

    def __init__(self, env, key='image'):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render('rgb_array')
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render('rgb_array')
        return obs



class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy='thread'):

        self._pickled_ctor = cloudpickle.dumps(constructor)  

        if strategy == 'process':
            import multiprocessing as mp
            context = mp.get_context('spawn')  

        elif strategy == 'thread':
            import multiprocessing.dummy as context  

        else:
            raise NotImplementedError(strategy)
        
        self._strategy = strategy

        
        self._conn, conn = context.Pipe()  

        self._process = context.Process(target=self._worker, args=(conn,))  

        atexit.register(self.close)  
        self._process.start()  

        self._receive()  

        self._obs_space = None
        self._act_space = None

    
    def access(self, name):
        self._conn.send((self._ACCESS, name))

        return self._receive  

    
    
    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    
    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass    # The connection was already closed.
        self._process.join(5)

    
    
    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access('obs_space')()
        return self._obs_space
    
    
    
    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access('act_space')()
        return self._act_space

    
    def step(self, action, blocking=False):
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    
    def reset(self, blocking=False):
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    
    def _receive(self):
        try:
            message, payload = self._conn.recv()  
        except (OSError, EOFError):
            raise RuntimeError('Lost connection to environment worker.')
        # Re-raise exceptions in the main process.
        
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        
        if message == self._RESULT:
            return payload
        raise KeyError('Received message of unexpected type {}'.format(message))

    
    def _worker(self, conn):
        try:

            
            
            ctor = cloudpickle.loads(self._pickled_ctor)  
            env = ctor()  

            conn.send((self._RESULT, None))    

            
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()  
                except (EOFError, KeyboardInterrupt):
                    break

                
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue

                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue

                
                if message == self._CLOSE:
                    break

                raise KeyError('Received message of unknown type {}'.format(message))
        
        
        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            print('Error in environment process: {}'.format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))

        
        finally:
            try:
                conn.close()
            except IOError:
                pass    # The connection was already closed.





def save_episode(directory, episode):
    episode = {key: convert(value) for key, value in episode.items()}
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = len(episode['action']) - 1
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    
    return value









