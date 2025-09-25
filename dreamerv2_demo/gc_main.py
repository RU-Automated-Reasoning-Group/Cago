import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from dreamerv2_demo import common
# import envs
import envs
import random
import numpy as np
import collections
import matplotlib.pyplot as plt
from dreamerv2_demo.common.replay import convert
import pathlib
import sys
import ruamel.yaml as yaml
import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import signal
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
from functools import partial
import pickle
from collections import defaultdict
from time import time
from tqdm import tqdm
import imageio
import numpy as np
import ruamel.yaml as yaml
import time
import warnings

import gc_agent
import common
import dreamerv2_demo.gc_goal_picker as gc_goal_picker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


class Method: 

    def __init__(self):
        self.cluster_ep_idx = 0
        self.if_train_goal_optimizer = False

    
    def Set_Config(self,):

        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent / 'Config/configs.yaml').read_text())

        parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)  

        config = common.Config(configs['defaults'])

        for name in parsed.configs:
            config = config.update(configs[name])
        config = old_config = common.Flags(config).parse(remaining)  

        logdir = pathlib.Path(config.logdir).expanduser()
        if logdir.exists():
            print('Loading existing config')
            yaml_config = yaml.safe_load((logdir / 'config.yaml').read_text())
            new_keys = ['train_seed_num', 'l2_threshold', 'if_image_obs','sample_range', 'count_lb', 'count_ub', 'if_goal_optimizer', 'goal_optimizer_start_step', 'jsrl_rollout_every', 'classifier', 'if_run_eagerly', 'if_modify_obs_dim', 'count_mode', 'l2_important_threshold', 'img_distance_threshold', 'important_dim_num', 'final_goal_sample_rate']
            for key in new_keys:
                if key not in yaml_config:
                    print(
                        f"key: {key} does not exist in saved config file, using default value from default config file")
                    yaml_config[key] = old_config[key]
            config = common.Config(yaml_config)
            config = common.Flags(config).parse(remaining)
            config.save(logdir / 'config.yaml')
            # config = common.Config(yaml_config)
            # config = common.Flags(config).parse(remaining)
        else:
            print('Creating new config')
            logdir.mkdir(parents=True, exist_ok=True)
            config.save(logdir / 'config.yaml')
        # print(config, '\n')
        # print('Logdir', logdir)

        return config


    
    def make_env(self, config, if_eval=False):

        
        def wrap_mega_env(e, info_to_obs_fn=None):
            e = common.GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
            if hasattr(e.act_space['action'], 'n'):
                e = common.OneHotAction(e)
            else:
                e = common.NormalizeAction(e)
            return e

        
        def wrap_lexa_env(e):
            e = common.GymWrapper(e)
            if hasattr(e.act_space['action'], 'n'):
                e = common.OneHotAction(e)
            else:
                e = common.NormalizeAction(e)
            # e = common.TimeLimit(e, 300)
            if if_eval:
                e = common.TimeLimit(e, 300)
            else:   
                e = common.TimeLimit(e, config.time_limit)
            return e

        
        if config.task in {'discwallsdemofetchpnp', 'wallsdemofetchpnp2', 'wallsdemofetchpnp3', 'demofetchpnp'}:

            from envs.customfetch.custom_fetch import DemoStackEnv, WallsDemoStackEnv, DiscreteWallsDemoStackEnv

            if 'walls' in config.task:
                if 'disc' in config.task:
                    env = DiscreteWallsDemoStackEnv(
                        max_step=config.time_limit, eval=if_eval, increment=0.01)
                else:
                    n = int(config.task[-1])
                    env = WallsDemoStackEnv(
                        max_step=config.time_limit, eval=if_eval, n=int(config.task[-1]))
            else:
                env = DemoStackEnv(max_step=config.time_limit, eval=if_eval)

            env = common.ConvertGoalEnvWrapper(env)

            # LEXA assumes information is in obs dict already, so move info dict into obs.
            info_to_obs = None

            
            def info_to_obs(info, obs):
                if info is None:
                    info = env.get_metrics_dict()
                obs = obs.copy()
                for k, v in info.items():
                    if eval:
                        if "metric" in k:
                            obs[k] = v
                    else:
                        if "above" in k:
                            obs[k] = v
                return obs

            

            class ClipObsWrapper:
                def __init__(self, env, obs_min, obs_max):
                    self._env = env
                    self.obs_min = obs_min
                    self.obs_max = obs_max

                def __getattr__(self, name):
                    return getattr(self._env, name)

                def step(self, action):
                    obs, rew, done, info = self._env.step(action)
                    new_obs = np.clip(obs['observation'],
                                    self.obs_min, self.obs_max)
                    obs['observation'] = new_obs
                    return obs, rew, done, info

            obs_min = np.ones(
                env.observation_space['observation'].shape) * -1e6  
            pos_min = [1.0, 0.3, 0.35]  # x, y, z
            if 'demofetchpnp' in config.task:
                obs_min[:3] = obs_min[5:8] = obs_min[8:11] = pos_min
                if env.n == 3:
                    obs_min[11:14] = pos_min

            obs_max = np.ones(
                env.observation_space['observation'].shape) * 1e6  
            pos_max = [1.6, 1.2, 1.0]  # x, y, z
            if 'demofetchpnp' in config.task:
                obs_max[:3] = obs_max[5:8] = obs_max[8:11] = pos_max
                if env.n == 3:
                    obs_max[11:14] = pos_max

            env = ClipObsWrapper(env, obs_min, obs_max)

            obs_min = np.concatenate(
                [env.workspace_min, [0., 0.], *[env.workspace_min for _ in range(env.n)]], 0)
            obs_max = np.concatenate(
                [env.workspace_max, [0.05, 0.05], *[env.workspace_max for _ in range(env.n)]], 0)

            env = common.NormObsWrapper(env, obs_min, obs_max)
            env = wrap_mega_env(env, info_to_obs)


        elif config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
            from envs.sibrivalry.ant_maze import AntMazeEnvFullDownscale, AntHardMazeEnvFullDownscale
            if 'hard' in config.task:
                env = AntHardMazeEnvFullDownscale(eval=if_eval)
            else:
                env = AntMazeEnvFullDownscale(eval=if_eval)
            env.max_steps = config.time_limit
            # Antmaze is a GoalEnv
            env = common.ConvertGoalEnvWrapper(env)
            info_to_obs = None
            if if_eval:
                def info_to_obs(info, obs):
                    if info is None:
                        info = env.get_metrics_dict()
                    obs = obs.copy()
                    for k,v in info.items():
                        if "metric" in k:
                            obs[k] = v
                    return obs
            env = wrap_mega_env(env, info_to_obs)


        elif 'pointmaze' in config.task:
            from envs.sibrivalry.toy_maze import MultiGoalPointMaze2D
            env = MultiGoalPointMaze2D(test=if_eval)
            env.max_steps = config.time_limit
            # PointMaze2D is a GoalEnv, so rename obs dict keys.
            env = common.ConvertGoalEnvWrapper(env)
            # LEXA assumes information is in obs dict already, so move info dict into obs.
            info_to_obs = None
            if if_eval:
                def info_to_obs(info, obs):
                    if info is None:
                        info = env.get_metrics_dict()
                    obs = obs.copy()
                    for k,v in info.items():
                        if "metric" in k:
                            obs[k] = v
                    return obs
            env = wrap_mega_env(env, info_to_obs)

            class GaussianActions:
                """Add gaussian noise to the actions.
                """
                def __init__(self, env, std):
                    self._env = env
                    self.std = std

                def __getattr__(self, name):
                    return getattr(self._env, name)

                def step(self, action):
                    new_action = action
                    if self.std > 0:
                        noise = np.random.normal(scale=self.std, size=2)
                        if isinstance(action, dict):
                            new_action = {'action': action['action'] + noise}
                        else:
                            new_action = action + noise

                    return self._env.step(new_action)
            env = GaussianActions(env, std=0)

            # obs = env.reset()


        elif 'dmc' in config.task:

            if if_eval:
                use_goal_idx=True
                log_per_goal=False

            else:
                use_goal_idx=False
                log_per_goal=True

            suite_task, obs = config.task.rsplit('_', 1)
            suite, task = suite_task.split('_', 1)
            if 'proprio' in config.task:
                env = envs.DmcStatesEnv(task, config.render_size, config.action_repeat, use_goal_idx, log_per_goal)
                if 'humanoid' in config.task:
                    keys = ['qpos', 'goal']
                    env = common.NormObsWrapper(env, env.obs_bounds[:, 0], env.obs_bounds[:, 1], keys)
            elif 'vision' in config.task:
                env = envs.DmcEnv(task, config.render_size, config.action_repeat, use_goal_idx, log_per_goal)

            env = wrap_lexa_env(env)


        elif config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or config.task == "PickCube-v0" or config.task == "PullCubeTool-v1" or config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1" or config.task == "PickAndPlace" or config.task == "FetchPush-v2" or config.task == "FetchSlide-v2" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task or "Adroit" in config.task or "meta" in config.task:
            
            
            import gymnasium as gym

            try:
                import mani_skill2.envs
            except:
                pass

            # from mani_skill2.utils.sapien_utils import vectorize_pose

            if config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or config.task == "PickCube-v0":

                demo_file_name= config.demo_path.split('/')[-1]
                controller = demo_file_name.split('.')[2]

                # supported controller modes: ['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 'pd_ee_delta_pose_align', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
                env = gym.make(config.task, obs_mode="state", control_mode=controller, render_mode="rgb_array", max_episode_steps=config.time_limit)

            if config.task == "PullCubeTool-v1":

                try:
                    import json
                    import mani_skill.envs
                    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
                except:
                    pass

                demo_file_name= config.demo_path.split('/')[-1]
                controller = demo_file_name.split('.')[2]

                task_demo_path = config.demo_path
                tra_json_path = task_demo_path.replace('.h5', '.json')
                with open(tra_json_path, 'r') as file:
                    data = json.load(file)

                # supported controller modes: ['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 'pd_ee_delta_pose_align', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
                env = gym.make(config.task, **data['env_info']['env_kwargs'], max_episode_steps=config.time_limit)
                env = CPUGymWrapper(env)

            elif config.task == 'OpenCabinetDrawer-v1':

                env = gym.make("OpenCabinetDrawer-v1", reward_mode = "sparse", obs_mode = "state", control_mode="base_pd_joint_vel_arm_pd_joint_vel", model_ids = ["1000"], render_mode="rgb_array", max_episode_steps=config.time_limit)
                
            elif config.task == "PushChair-v1":

                env = gym.make("PushChair-v1", reward_mode = "sparse", obs_mode = "state", model_ids = ["3001"], render_mode="rgb_array", max_episode_steps=config.time_limit)          

            elif config.task == "PickAndPlace":
                env = gym.make('FetchPickAndPlace-v2', render_mode="rgb_array", max_episode_steps=config.time_limit)

            elif "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:
                env = gym.make(config.task,  render_mode="rgb_array", max_episode_steps=config.time_limit)
            
            elif config.task == "FetchPush-v2" or config.task == "FetchSlide-v2":
                env = gym.make(config.task,  render_mode="rgb_array", max_episode_steps=config.time_limit)
            
            elif "Adroit" in config.task:
                env = gym.make(config.task,  render_mode="rgb_array", max_episode_steps=config.time_limit)
            
            elif "meta" in config.task:

                env = gym.make(config.task, max_episode_steps=config.time_limit, generate_image=False, sparse = True)
            
            # obs = env.reset() # reset with a seed for randomness

            # print("Observation space", env.observation_space)
            # print("Action space", env.action_space)

            # # print(vectorize_pose(env.box_hole_pose))
            # # state = env.get_state()
            # print(obs)
            # # print('=============')
            # # print(state)
            # terminated, truncated = False, False
            # while not terminated and not truncated:
            #     action = env.action_space.sample()
            #     print(action)
            #     obs, reward, terminated, truncated, info = env.step(action)
            # env.close()

            # sys.exit()


            def gymnasium_env(e):

                # print(e.obs_space)

                if config.if_use_demo:

                    e = common.GymnasiumWrapper_Demo(e, if_eval=if_eval, reset_with_seed=if_eval, config=config)  # assign seed and obs space = goal space

                else:

                    if config.if_env_random_reset:
                        e = common.GymnasiumWrapper_1(e, if_eval=if_eval)  # random seed
                    else:
                        e = common.GymnasiumWrapper_0(e, if_eval=if_eval, reset_with_seed=if_eval)  # assign seed

                
                # Action Wrapper
                if hasattr(e.act_space['action'], 'n'):
                    e = common.OneHotAction(e)
                else:
                    e = common.NormalizeAction(e)

                # print(e.obs_space)
                # e = common.TimeLimit(e, config.time_limit)

                # print(e.obs_space)
                return e
            
            env = gymnasium_env(env)

            # obs = env.reset()

            # print(env.obs_space)
            # print("Observation space", env.observation_space)
            # print("Action space", env.action_space)


        else:
            raise NotImplementedError

        return env
    
    
    
    def make_images_render_fn(self, config):

        images_render_fn = None

        if 'demofetchpnp' in config.task:

            def images_render_fn(env, obs_list):

                sim = env.sim
                all_img = []
                env.reset()
                # move the robot arm out of the way
                if env.n == 3:
                    out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06, 1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                    -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                    2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                    1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                    -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                    8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                    1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                    4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                    -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                    5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                    -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                    -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                    -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                    -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                    1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                    -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                    -5.58792469e-06, -2.07082526e-17])

                sim.set_state_from_flattened(out_of_way_state)
                sim.forward()

                # inner_env.goal = ep['goal'][0]
                # now render the states.
                sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                site_id = sim.model.site_name2id('gripper_site')

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)
                
                for obs in [*obs_list]:

                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
                    # set the end effector site instead of the actual end effector.
                    sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
                    # set the objects
                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1, 0, 0, 0]])

                    sim.forward()
                    img = sim.render(height=200, width=200, camera_name="external_camera_0")[::-1]
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis = 1)
                all_imgs = np.expand_dims(all_imgs, axis = 0)

                # logger.image(log_label, all_imgs)
        
                return all_imgs


        elif config.task in {'umazefull','umazefulldownscale', 'hardumazefulldownscale'}:

            def images_render_fn(env, obs_list):

                ant_env = env.maze.wrapped_env

                inner_env = env._env._env._env

                all_img = []
                for obs in [*obs_list]:
                    inner_env.maze.wrapped_env.set_state(obs[:15], np.zeros_like(obs[:14]))
                    inner_env.maze.wrapped_env.sim.forward()
                    img = env.render(mode='rgb_array')
                    
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis = 1)
                all_imgs = np.expand_dims(all_imgs, axis = 0)

                # logger.image(log_label, all_imgs)
        
                return all_imgs
        

        elif 'pointmaze' in config.task:


            def images_render_fn(env, obs_list):

                all_img = []
                inner_env = env._env._env._env._env

                for xy in [*obs_list]:

                    inner_env.g_xy = xy
                    inner_env.s_xy = xy
                    img = env.render()
                    all_img.append(img)

                env.clear_plots()

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis= 1 )
                all_imgs = np.expand_dims(all_imgs, axis = 0)

                # logger.image(log_label, all_imgs)
        
                return all_imgs


        elif 'dmc_walker_walk_proprio' == config.task:

            def images_render_fn(env, obs_list):

                all_img = []
                inner_env = env._env._env._env._env
                for qpos in [*obs_list]:
                    size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()
                    img = env.render()
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis= 1 )
                all_imgs = np.expand_dims(all_imgs, axis = 0)

                # logger.image(log_label, all_imgs)
        
                return all_imgs


        elif config.task == "PickAndPlace" or config.task == "FetchPush-v2" or config.task == "FetchSlide-v2":

            def images_render_fn(env, obs_list):

                all_img = []

                for obs in [*obs_list]:

                    img = env.render_with_obs(obs, obs[3:6], width=200, height=200)
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                # logger.image(log_label, all_imgs)
        
                return all_imgs


        elif  "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            def images_render_fn(env, obs_list):

                all_img = []

                for obs in [*obs_list]:

                    if "HandManipulateBlockRotateXYZ" in config.task or "HandManipulatePenRotate" in config.task:
                        img = env.render_with_obs(obs, obs[-4:], width=200, height=200)
                    else:
                        img = env.render_with_obs(obs, obs[-7:], width=200, height=200)
                    all_img.append(img)

                all_imgs = np.stack(all_img[1:], 0)

                all_imgs = np.array(all_imgs)

                all_imgs = np.concatenate(all_imgs, axis=1)
                all_imgs = np.expand_dims(all_imgs, axis=0)

                # logger.image(log_label, all_imgs)
        
                return all_imgs            


        return images_render_fn


    
    def make_ep_render_fn(self, config):
        
        episode_render_fn = None
        if config.no_render:
            return episode_render_fn

        if 'demofetchpnp' in config.task:
            import cv2

            def episode_render_fn_original(env, ep):
                sim = env.sim
                all_img = []
                # reset the robot.
                env.reset()
                inner_env = env._env._env._env._env._env  
                # move the robot arm out of the way
                if env.n == 3:
                    out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06,
                                                1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                -5.58792469e-06, -2.07082526e-17])

                sim.set_state_from_flattened(out_of_way_state)
                sim.forward()
                inner_env.goal = ep['goal'][0]
                subgoal_time = ep['log_subgoal_time']
                sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                site_id = sim.model.site_name2id('gripper_site')

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)
                
                for i, obs in enumerate(ep['observation']):
                    # print("obs", obs)
                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
                    # set the end effector site instead of the actual end effector.
                    sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
                    # set the objects
                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [
                                                *pos, *[1, 0, 0, 0]])

                    sim.forward()
                    img = sim.render(height=200, width=200,
                                    camera_name="external_camera_0")[::-1]
                    if subgoal_time > 0 and i >= subgoal_time:
                        img = img.copy()
                        cv2.putText(
                            img,
                            f"expl",
                            (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    all_img.append(img)
                all_img = np.stack(all_img, 0)
                return all_img
            
            def episode_render_fn(env, ep, if_eval=False):
                sim = env.sim
                all_img = []
                goals = []
                executions = []
                env.reset()
                # move the robot arm out of the way
                if env.n == 3:
                    out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06, 1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                    -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                    2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                    1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                    -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                    8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                    1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                    4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                    -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                    5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                    -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                    -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                    -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                    -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                    1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                    -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                    -5.58792469e-06, -2.07082526e-17])

                sim.set_state_from_flattened(out_of_way_state)
                sim.forward()

                # inner_env.goal = ep['goal'][0]
                # now render the states.
                sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                site_id = sim.model.site_name2id('gripper_site')

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max - env.obs_min)

                for goal, obs in zip(ep['goal'], ep['observation']):

                    # render obs img
                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
                    # set the end effector site instead of the actual end effector.
                    sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
                    # set the objects
                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1, 0, 0, 0]])

                    sim.forward()
                    img = sim.render(height=200, width=200, camera_name="external_camera_0")[::-1]

                    # render goal img
                    goal = unnorm_ob(goal)
                    grip_pos = goal[:3]
                    gripper_state = goal[3:5]
                    all_obj_pos = np.split(goal[5:5+3*env.n], env.n)
                    # set the end effector site instead of the actual end effector.
                    sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
                    # set the objects
                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1, 0, 0, 0]])

                    sim.forward()
                    goal_img = sim.render(height=200, width=200, camera_name="external_camera_0")[::-1]

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)


                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img

        elif config.task in {'umazefull','umazefulldownscale', 'hardumazefulldownscale'}:

            def episode_render_fn(env, ep, if_eval=False):

                ant_env = env.maze.wrapped_env

                ant_env.set_state(ep['goal'][0][:15], ep['goal'][0][:14])

                inner_env = env._env._env._env
                all_img = []
                for obs, goal in zip(ep['observation'], ep['goal']):
                    inner_env.maze.wrapped_env.set_state(obs[:15], np.zeros_like(obs[:14]))
                    inner_env.g_xy = goal[:2]
                    inner_env.maze.wrapped_env.sim.forward()
                    img = env.render(mode='rgb_array')
                    

                    inner_env.maze.wrapped_env.set_state(goal[:15], goal[:14])
                    inner_env.g_xy = goal[:2]
                    ant_env.sim.forward()
                    goal_img = env.render(mode='rgb_array')

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img
        
        elif 'pointmaze' in config.task:
            def episode_render_fn(env, ep, if_eval=False):
                all_img = []
                inner_env = env._env._env._env._env
                for g_xy, xy in zip(ep['goal'], ep['observation']):
                    inner_env.g_xy = g_xy
                    inner_env.s_xy = xy
                    img = env.render()
                    all_img.append(img)
                env.clear_plots()

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img
        
        # our walker
        elif 'dmc_walker_walk_proprio' == config.task:

            def episode_render_fn(env, ep, if_eval=False):
                all_img = []
                inner_env = env._env._env._env._env

                for qpos, goal in zip(ep['qpos'], ep['goal']):
                    size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()
                    img = env.render()

                    size = inner_env.physics.get_state().shape[0] - goal.shape[0]
                    inner_env.physics.set_state(np.concatenate((goal, np.zeros([size]))))
                    inner_env.physics.step()
                    goal_img = env.render()

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]
                
                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img

        elif 'dmc_humanoid_walk_proprio' == config.task:

            def episode_render_fn(env, ep, if_eval=False):
                all_img = []
                inner_env = env._env._env._env._env._env
                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max -    env.obs_min)
                for qpos in ep['qpos']:
                    size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
                    qpos = unnorm_ob(qpos)
                    inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()
                    img = env.render()
                    all_img.append(img)
                    
                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]
                

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img

        elif config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or config.task == "PickCube-v0" or config.task == "PullCubeTool-v1" or config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1":

            def episode_render_fn(env, ep, if_eval=False):

                
                all_img = []

                # print(ep['goal'][0])
                # if env._env.spec.id == "PullCubeTool-v1" and if_eval:
                #     env.reset()
                #     initial_image = env.render()
                #     imageio.imsave(f"initial_img_{env.goal_idx}.png", initial_image)


                goal_render_state = env.get_demogoal_render_state(env.goal_idx, ep['goal'][0])

            
                if goal_render_state is None:
                    goal_img = None
                else:
                    env.unwrapped.set_state(goal_render_state)
                    goal_img = env.render()

                
                # imageio.imsave(f"goal_img_{env.goal_idx}.png", goal_img)
                
                # print("goal_render_state:", goal_render_state)

                for env_state in ep['env_states']:
                    env.unwrapped.set_state(env_state)
                    img = env.render()

                    if goal_img is not None:
                        image = np.concatenate([goal_img, img], -3)
                    else:
                        image = img

                    all_img.append(image)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]
                
                # imageio.mimsave(f"ep_img_{env.goal_idx}.gif", ep_img)
                
                # imageio.imsave(f"ep_img_{env.goal_idx}.png", image)

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                
                # imageio.mimsave(f"ep_render_{env.goal_idx}.gif", ep_img)

                return ep_img

        elif "meta" in config.task:

            def episode_render_fn(env, ep, if_eval=False):

                
                all_img = []

                # print(ep['goal'][0])

                goal_render_state = env.get_demogoal_render_state(env.goal_idx, ep['goal'][0])

                if goal_render_state is None:
                    goal_img = None
                else:
                    env.unwrapped.set_env_state(goal_render_state)
                    goal_img = env.render()
                
                # print("goal_render_state:", goal_render_state)

                for env_state in ep['env_states']:
                    env.unwrapped.set_env_state(env_state)
                    img = env.render()

                    if goal_img is not None:
                        image = np.concatenate([goal_img, img], -3)
                    else:
                        image = img

                    all_img.append(image)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]
                

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                # ============================================================================================================debug
                # goal_idx = 0
                # seed = goal_idx
                # save_dir = "/common/users/yd374/ach/ACH_Server/Experiments/"
                # env.goal_rate = 1
                # env.set_goal_idx(goal_idx)
                # obs = env.reset()
                # done = False
                # save_gif = True
                # policy = None
                # traj_data = defaultdict(list)
                # frames = []
                # state_frames = []

                # while not done:
                #     # Record frames for GIF if enabled
                #     if save_gif:
                #         frame = env.render()
                #         frames.append(frame)

                #     # Use the policy to select an action
                #     if policy is None:
                #         action = env.action_space.sample()  # Random policy
                #         action = {"action": action}
                #     else:
                #         action = policy(obs)[0]
                #         action = {"action": action}
                    
                #     # Take a step in the environment
                #     next_obs = env.step(action)

                #     state = env.unwrapped.get_env_state()
                #     done = next_obs["is_last"]
                #     next_obs = next_obs["observation"]


                #     # Save data
                #     traj_data["actions"].append(action["action"].tolist())
                #     traj_data["observations"].append(next_obs.tolist())
                #     traj_data["env_states"].append(state)

                #     obs = next_obs

                # # Test the state rendering
                # for state in traj_data["env_states"]:

                #     env.unwrapped.set_env_state(state)
                #     state_frames.append(env.render())

                # # Optionally save GIF for each trajectory
                # if save_gif:
                #     gif_path = os.path.join(save_dir, f"trajectory_seed_{seed}.gif")
                #     imageio.mimsave(gif_path, frames, fps=30)
                #     print(f"Saved GIF for seed {seed}: {gif_path}, length: {len(frames)}")

                #     state_gif_path = os.path.join(save_dir, f"state_trajectory_seed_{seed}.gif")
                #     imageio.mimsave(state_gif_path, state_frames, fps=30)

                # demo_tra = env.all_demo_trajectories[goal_idx]

                # demo_state_frames = []
                # for state in demo_tra["env_states"]:

                #     env.unwrapped.set_env_state(state)
                #     demo_state_frames.append(env.render())

                # demo_gif_path = os.path.join(save_dir, f"demo_trajectory_seed_{seed}_D.gif")
                # imageio.mimsave(demo_gif_path, demo_state_frames, fps=30)

                # ============================================================================================================debug

                return ep_img

        elif "Adroit" in config.task:

            def episode_render_fn(env, ep, if_eval=False):

                
                all_img = []

                # print(ep['goal'][0])

                goal_render_state = env.get_demogoal_render_state(env.goal_idx, ep['goal'][0])

                if goal_render_state is None:
                    goal_img = None
                else:
                    # print("goal_render_state:", goal_render_state)
                    env.unwrapped.set_env_state(goal_render_state)
                    goal_img = env.render()
                
                # print("goal_render_state:", goal_render_state)

                for env_state in ep['env_states']:
                    env.unwrapped.set_env_state(env_state.item())  
                    img = env.render()

                    if goal_img is not None:
                        image = np.concatenate([goal_img, img], -3)
                    else:
                        image = img

                    all_img.append(image)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]
                

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img
      
        elif config.task == "PickAndPlace" or config.task == "FetchPush-v2" or config.task == "FetchSlide-v2" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:

            def episode_render_fn_original(env, ep):

                # goal_idx = ep['goal_idx'][0]
                # env._env.set_goal_idx(goal_idx)

                env.reset()
                
                all_img = []

                new_ep = []

                for action in ep['action']:

                    # action = env.action_space.sample()  # Random policy
                    action = {'action': action}
                    
                    obs = env.step(action)

                    new_ep.append(obs)

                    img = env.render()

                    # imageio.imsave('step.png', img)

                    all_img.append(env.render())

                ep_img = np.stack(all_img, 0)

                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img
            
            def episode_render_fn(env, ep, if_eval=False):

                env.reset()
                
                all_img = []

                for obs, goal in zip(ep['observation'], ep['desired_goal']):

                    img = env.render_with_obs(obs, goal, width=200, height=200)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img
            
            def episode_render_fn_with_demo(env, ep):

                # env.reset()
                
                all_img = []

                goal_img = env.render_with_obs(ep['goal'][0], ep['desired_goal'][0], width=200, height=200)

                for obs in ep['observation']:

                    img = env.render_with_obs(obs, ep['desired_goal'][0], width=200, height=200)

                    img = np.concatenate([goal_img, img], -3)

                    all_img.append(img)

                ep_img = np.stack(all_img, 0)
                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img
            
            if self.config.if_use_demo:

                return episode_render_fn_with_demo
   
        return episode_render_fn


    
    def make_eval_fn(self, config):


        episode_render_fn = self.make_ep_render_fn(config)

        # 3-block
        if 'demofetchpnp' in config.task:

            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]

                # eval_goal_idxs = range(24, 36)
                # TODO: revert back to 3 block goals
                eval_goal_idxs = range(len(env.get_goals()))

                num_eval_eps = 10
                all_metric_success = []
                # key is metric name, value is list of size num_eval_eps
                ep_metrics = collections.defaultdict(list)

                all_ep_videos = []

                
                for ep_idx in range(num_eval_eps):

                    
                    should_video = ep_idx == 0 and episode_render_fn is not None

                    #debug
                    # should_video = True
                    # executions = []
                    # goals = []

                    for idx in eval_goal_idxs:
                        driver.reset()
                        env.set_goal_idx(idx)

                        #debug
                        # if idx == 1:
                        #     continue

                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0]  # get episode data of 1st env.

                        ep = {k: driver._convert(
                            [t[k] for t in ep]) for k in ep[0]}
                        score = float(ep['reward'].astype(np.float64).sum())
                        print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
                        # render the goal img and rollout
                        
                        for k, v in ep.items():
                            if 'metric_success/goal_' in k:
                                ep_metrics[k].append(np.max(v))
                                all_metric_success.append(np.max(v))

                        if should_video:
                            # render the goal img and rollout
                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    
                    if should_video:
                        # num_goals x T x H x W x C
                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                
                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all',
                              all_metric_success)
                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))
                logger.write()  


            def eval_from_specific_goal(driver, eval_policy, logger):

                save_path = str(config.logdir) + "/ep_videos_from_specific_goal/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                env = driver._envs[0]

                # eval_goal_idxs = range(24, 36)
                # TODO: revert back to 3 block goals
                eval_goal_idxs = range(len(env.get_goals()))

                goal_list = env.get_goals()
                num_eval_eps = 1
                all_metric_success = []
                # key is metric name, value is list of size num_eval_eps
                ep_metrics = collections.defaultdict(list)

                def set_state(env, obs):

                    sim = env.sim
                    # move the robot arm out of the way
                    if env.n == 3:
                        out_of_way_state = np.array([4.40000000e+00,    4.04999349e-01,    4.79999636e-01,    2.79652104e-06, 1.56722299e-02, -3.41500342e+00, 9.11469058e-02, -1.27681180e+00,
                                                        -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
                                                        2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
                                                        1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
                                                        -2.28652597e-07, 2.56090909e-07, -1.20181003e-15, 1.32999955e+00,
                                                        8.49999274e-01, 4.24784489e-01, 1.00000000e+00, -2.77140579e-07,
                                                        1.72443027e-07, -1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
                                                        4.24784489e-01, 1.00000000e+00, -2.31485576e-07, 2.31485577e-07,
                                                        -6.68816586e-16, -4.48284993e-08, -8.37398903e-09, 7.56100615e-07,
                                                        5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
                                                        -7.15601860e-02, -9.44665089e-02, 1.49646097e-02, -1.10990294e-01,
                                                        -3.30174644e-03, 1.19462201e-01, 4.05130821e-04, -3.95036450e-04,
                                                        -1.53880539e-07, -1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
                                                        -6.18188284e-06, 1.31307184e-17, -1.03617993e-07, -1.66528917e-07,
                                                        1.06089030e-14, 6.69000941e-06, -4.16267252e-06, 3.63225324e-17,
                                                        -1.39095626e-07, -1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
                                                        -5.58792469e-06, -2.07082526e-17])

                    # sim.set_state_from_flattened(out_of_way_state)
                    # sim.forward()

                    # inner_env.goal = ep['goal'][0]
                    # now render the states.
                    sites_offset = (sim.data.site_xpos - sim.model.site_pos)
                    site_id = sim.model.site_name2id('gripper_site')

                    def unnorm_ob(ob):
                        return env.obs_min + ob * (env.obs_max - env.obs_min)

                    # render obs img
                    obs = unnorm_ob(obs)
                    grip_pos = obs[:3]
                    gripper_state = obs[3:5]
                    all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
                    # set the end effector site instead of the actual end effector.
                    sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
                    # set the objects
                    for i, pos in enumerate(all_obj_pos):
                        sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1, 0, 0, 0]])

                    sim.forward()

                driver.set_state_fn = set_state
                

                ep_num = 0
                for ep_idx in range(num_eval_eps):

                    
                    should_video = True

                    #debug
                    # should_video = True
                    # executions = []
                    # goals = []

                    for start_goal_idx in eval_goal_idxs:

                        start_goal = goal_list[start_goal_idx]

                        for idx in eval_goal_idxs:

                            if idx == start_goal_idx:
                                continue

                            driver.reset()

                            env.set_goal_idx(idx)
                            driver.if_set_initial_state = True
                            driver.initial_state = start_goal

                            driver(eval_policy, episodes=1)
                            """ rendering based on state."""
                            ep = driver._eps[0]  # get episode data of 1st env.

                            ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                            score = float(ep['reward'].astype(np.float64).sum())

                            for k, v in ep.items():
                                if 'metric_success/goal' in k:
                                    ep_metrics[k].append(np.max(v))
                                    all_metric_success.append(np.max(v))

                            if should_video:
                                # render the goal img and rollout
                                ep_video = episode_render_fn(env, ep)

                                imageio.mimsave(save_path + f'ep_with_start_goal{start_goal_idx}_end_goal{idx}.gif', ep_video)

                            ep_num += 1

                
                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success_from_specific_goal', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_end_goal_{key}', np.mean(value))

                logger.write()  

            # return eval_from_specific_goal


        elif config.task in {'umazefull','umazefulldownscale', 'hardumazefulldownscale'}:
            
            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals()) if len(env.get_goals()) > 0 else 5 # MEGA uses 30 episodes for eval.
                num_eval_eps = 10
                executions = []
                goals = []
                all_metric_success = []
                all_ep_videos = []
                # key is metric name, value is list of size num_eval_eps
                ep_metrics = collections.defaultdict(list)
                for ep_idx in range(num_eval_eps):
                    should_video = ep_idx == 0 and episode_render_fn is not None
                    for idx in range(num_goals):
                        env.set_goal_idx(idx)
                        driver.reset()

                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0] # get episode data of 1st env.
                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}

                        score = float(ep['reward'].astype(np.float64).sum())
                        print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
                        # render the goal img and rollout

                        for k, v in ep.items():
                            if 'metric' in k:
                                ep_metrics[k].append(np.max(v))
                                all_metric_success.append(np.max(v))

                        if should_video:
                            # render the goal img and rollout
                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    
                    if should_video:
                        # num_goals x T x H x W x C
                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all', all_metric_success)
                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))
                logger.write()
            
            def eval_fn_collect(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals()) if len(env.get_goals()) > 0 else 5 # MEGA uses 30 episodes for eval.
                num_eval_eps = 10
                executions = []
                goals = []
                all_metric_success = []
                # key is metric name, value is list of size num_eval_eps
                ep_metrics = collections.defaultdict(list)

                print("==============================collect the demonstration data=======================================")
                
                for idx in range(num_goals):

                    if idx < 15:

                        num_eval_eps = 50

                    else:
                        num_eval_eps = 100
                        
                    for ep_idx in range(num_eval_eps):

                        env.set_goal_idx(idx)

                        driver.reset()

                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0] # get episode data of 1st env.
                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                        score = float(ep['reward'].astype(np.float64).sum())
                        print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
                        # render the goal img and rollout

                        for k, v in ep.items():
                            if 'metric' in k:
                                ep_metrics[k].append(np.max(v))
                                all_metric_success.append(np.max(v))

                                if np.max(v) > 0:
                                    np.savez(f'/common/home/yd374/ACH_Server/Experiment/Ant_Maze_Demo/goal_{idx}_ep_{ep_idx}.npz', observation=ep['observation'], action=ep['action'])
                                else:
                                    ep_idx -= 1

                all_metric_success = np.mean(all_metric_success)

                for key, value in ep_metrics.items():
                    print(f'mean_eval_{key}', np.mean(value))

                sys.exit()

            def eval_from_specific_goal(driver, eval_policy, logger):

                save_path = str(config.logdir) + "/ep_videos_from_specific_goal/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                env = driver._envs[0]

                # eval_goal_idxs = range(24, 36)
                # TODO: revert back to 3 block goals
                # eval_goal_idxs = range(len(env.get_goals()))

                eval_goal_idxs = [i * 4 for i in range(int(len(env.get_goals())/4))]

                goal_list = env.get_goals()
                num_eval_eps = 1
                all_metric_success = []
                # key is metric name, value is list of size num_eval_eps
                ep_metrics = collections.defaultdict(list)

                def set_state(env, obs):
                    ant_env = env.maze.wrapped_env

                    ant_env.set_state(obs[:15], obs[:14])

                    inner_env = env._env._env._env

                    inner_env.maze.wrapped_env.set_state(obs[:15], np.zeros_like(obs[:14]))
                    inner_env.maze.wrapped_env.sim.forward()


                driver.set_state_fn = set_state
                

                ep_num = 0
                for ep_idx in range(num_eval_eps):

                    
                    should_video = True

                    #debug
                    # should_video = True
                    # executions = []
                    # goals = []

                    for start_goal_idx in eval_goal_idxs:

                        start_goal = goal_list[start_goal_idx]

                        for idx in eval_goal_idxs:

                            if idx == start_goal_idx:
                                continue

                            driver.reset()

                            env.set_goal_idx(idx)
                            driver.if_set_initial_state = True
                            driver.initial_state = start_goal

                            driver(eval_policy, episodes=1)
                            """ rendering based on state."""
                            ep = driver._eps[0]  # get episode data of 1st env.

                            ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                            score = float(ep['reward'].astype(np.float64).sum())

                            for k, v in ep.items():
                                if 'metric' in k:
                                    ep_metrics[k].append(np.max(v))
                                    all_metric_success.append(np.max(v))

                            if should_video:
                                # render the goal img and rollout
                                # print(ep['observation'], ep['goal'])
                                ep_video = episode_render_fn(env, ep)

                                imageio.mimsave(save_path + f'ep_with_start_goal{start_goal_idx}_end_goal{idx}.gif', ep_video)

                            ep_num += 1

                
                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success_from_specific_goal', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_end_goal{key}', np.mean(value))

                logger.write()  

            # return eval_from_specific_goal


        elif 'pointmaze' in config.task:
            
            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals()) # maze has 5 goals
                num_eval_eps = 10
                all_ep_videos = []
                all_metric_success = []
                all_metric_success_cell = []
                ep_metrics = collections.defaultdict(list)
                for ep_idx in range(num_eval_eps):
                    should_video = ep_idx == 0 and episode_render_fn is not None
                    for idx in range(num_goals):
                        env.set_goal_idx(idx)
                        driver(eval_policy, episodes=1)
                        """ rendering based on state."""
                        ep = driver._eps[0] # get episode data of 1st env.
                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                        # aggregate goal metrics across goals together.
                        for k, v in ep.items():
                            if 'metric' in k:

                                ep_metrics[k].append(np.max(v))

                                if 'cell' in k.split('/')[0]:
                                    all_metric_success_cell.append(np.max(v))
                                else:
                                    all_metric_success.append(np.max(v))

                        if should_video:
                            # render the goal img and rollout
                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    
                    if should_video:
                        # num_goals x T x H x W x C
                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all', all_metric_success)
                all_metric_success_cell = np.mean(all_metric_success_cell)
                logger.scalar('mean_eval_metric_success_cell/goal_all', all_metric_success_cell)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))
                
                logger.write()


        elif 'dmc' in config.task:
       
            def eval_fn(driver, eval_policy, logger):
                env = driver._envs[0]
                num_goals = len(env.get_goals())
                num_eval_eps = 10
                all_ep_videos = []
                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                ep_metrics = collections.defaultdict(list)
                for ep_idx in range(num_eval_eps):
                    should_video = ep_idx == 0 and episode_render_fn is not None
                    for idx in range(num_goals):
                        env.set_goal_idx(idx)
                        env.reset()
                        driver(eval_policy, episodes=1)
                        ep = driver._eps[0] # get episode data of 1st env.
                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                        score = float(ep['reward'].astype(np.float64).sum())
                        print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
                        for k, v in ep.items():
                            if 'metric_success' in k:
                                all_metric_success.append(np.max(v))
                                ep_metrics[k].append(np.max(v))
                            elif 'metric_reward' in k:
                                ep_metrics[k].append(np.sum(v))

                        if should_video:
                            # render the goal img and rollout
                            ep_video = episode_render_fn(env, ep)
                            all_ep_videos.append(ep_video[None])

                    
                    if should_video:
                        # num_goals x T x H x W x C
                        gc_video = np.concatenate(all_ep_videos, 0)

                        logger.video(f'eval_gc_policy{ep_idx}', gc_video)

                # collect all the goal success metrics and get average
                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success/goal_all', all_metric_success)
                for key, value in ep_metrics.items():
                    if 'metric_success' in key:
                        logger.scalar(f'mean_eval_{key}', np.mean(value))
                    elif 'metric_reward' in key:
                        logger.scalar(f'sum_eval_{key}', np.mean(value))
                logger.write()

            def eval_from_specific_goal(driver, eval_policy, logger):

                save_path = str(config.logdir) + "/ep_videos_from_specific_goal/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                env = driver._envs[0]

                # eval_goal_idxs = range(24, 36)
                # TODO: revert back to 3 block goals
                # eval_goal_idxs = range(len(env.get_goals()))

                eval_goal_idxs = range(len(env.get_goals()))

                goal_list = env.get_goals()
                num_eval_eps = 1
                all_metric_success = []
                # key is metric name, value is list of size num_eval_eps
                ep_metrics = collections.defaultdict(list)

                def set_state(env, qpos):
                    inner_env = env._env._env._env._env

                    size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
                    inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
                    inner_env.physics.step()


                driver.set_state_fn = set_state
                

                ep_num = 0
                for ep_idx in range(num_eval_eps):

                    
                    should_video = True

                    #debug
                    # should_video = True
                    # executions = []
                    # goals = []

                    for start_goal_idx in eval_goal_idxs:

                        start_goal = goal_list[start_goal_idx]

                        for idx in eval_goal_idxs:

                            if idx == start_goal_idx:
                                continue

                            driver.reset()

                            env.set_goal_idx(idx)
                            driver.if_set_initial_state = True
                            driver.initial_state = start_goal

                            driver(eval_policy, episodes=1)
                            """ rendering based on state."""
                            ep = driver._eps[0]  # get episode data of 1st env.

                            ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                            score = float(ep['reward'].astype(np.float64).sum())

                            for k, v in ep.items():
                                if 'metric_success' in k:
                                    ep_metrics[k].append(np.max(v))
                                    all_metric_success.append(np.max(v))

                            if should_video:
                                # render the goal img and rollout
                                # print(ep['observation'], ep['goal'])
                                ep_video = episode_render_fn(env, ep)

                                imageio.mimsave(save_path + f'ep_with_start_goal{start_goal_idx}_end_goal{idx}.gif', ep_video)

                            ep_num += 1

                
                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success_from_specific_goal', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_end_goal{key}', np.mean(value))

                logger.write()  

            # return eval_from_specific_goal


        elif config.task == "PickAndPlace" or config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or config.task == "PickCube-v0" or config.task == "PullCubeTool-v1" or "Adroit" in config.task or config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1" or "meta" in config.task:


            def eval_fn_original(driver, eval_policy, logger):

                env = driver._envs[0]
                num_eval_eps = 10
                executions = []
                goals = []
                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []

                all_ep_video = []

                for ep_idx in range(num_eval_eps):

                    obs = env.reset()

                    for k in obs:
                        obs[k] = np.stack([obs[k]])
                    
                    state = None
                    done = False

                    ep = []
                    ep_video = []
                    while not done:

                        # print(obs)
                        actions, state = eval_policy(obs, state)

                        actions = [{k: np.array(actions[k][0]) for k in actions}]

                        obs = env.step(actions[0])

                        for k in obs:
                            obs[k] = np.stack([obs[k]])

                        done = obs['is_last']

                        ep.append(obs)

                        ep_video.append(env.render())


                    ep_video = np.stack(ep_video, 0)  
                    all_ep_video.append(ep_video[None])  # 1 x T x H x W x C

                    ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                    all_metric_success.append(max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0) # ep_num x T x H x W x C

                logger.video(f'eval_gc_policy', all_ep_video)


                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_metric_success', all_metric_success)

                logger.write()

            
            
            def eval_fn_with_demo_seed_reset(driver, eval_policy, logger):
                env = driver._envs[0]

                env = env._env
                eval_goal_num = len(env.seed_list)
                num_eval_eps = 1
                train_seed_all_ep_video = []
                all_ep_video = []
                goals = []
                # key is metric name, value is list of size num_eval_eps

                train_seed_all_metric_success = []
                train_seed_all_env_original_success = []
                train_seed_all_distance_from_goal = []
                train_seed_all_distance_from_goal_object = []
                all_metric_success = []
                all_env_original_success = []
                all_distance_from_goal = []
                all_distance_from_goal_object = []
                ep_metrics = collections.defaultdict(list)
                
                if eval_goal_num > 50:
                    random_goal_indices = np.random.choice(eval_goal_num, size=50, replace=False)
                else:
                    random_goal_indices = list(range(eval_goal_num))  
                    random.shuffle(random_goal_indices) 

            

                
                # train_seed_list = env.train_seed_lists
                for goal_idx in range(env.train_seed_num):

                    for ep_idx in range(num_eval_eps):

                        should_video = episode_render_fn is not None and ep_idx == 0

                        env.goal_rate = 1
                        env.set_goal_idx(goal_idx)

                        driver(eval_policy, episodes=1)
                        ep = driver._eps[0] # get episode data of 1st env.
                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                        # score = float(ep['reward'].astype(np.float64).sum())


                        train_seed_all_metric_success.append(np.max(ep['is_rate_obs_success']))
                        train_seed_all_env_original_success.append(np.max(ep['is_terminal']))

                        if ep['observation'][-1].shape[0] == ep['goal'][-1].shape[0]:
                            distance_from_goal = np.linalg.norm(ep['observation'][-1] - ep['goal'][-1])
                            train_seed_all_distance_from_goal.append(distance_from_goal)

                            if config.task == "PegInsertionSide-v0":
                                distance_from_goal_object = np.linalg.norm(env.obs2goal(ep['observation'][-1]) - env.obs2goal(ep['goal'][-1]))
                                train_seed_all_distance_from_goal_object.append(distance_from_goal_object)


                        if should_video:
                            """ rendering based on state."""
                            # render the goal img and rollout
                            ep_video = episode_render_fn(env, ep, if_eval=True)
                            
                            # imageio.mimsave(f'ep_{goal_idx}.gif', ep_video)
                            train_seed_all_ep_video.append(ep_video[None])  # 1 x T x H x W x C


                i = 0

                
                for goal_idx in random_goal_indices:

                    i += 1
                    for ep_idx in range(num_eval_eps):

                        should_video = episode_render_fn is not None and ep_idx == 0 and i <= 5

                        env.goal_rate = 1
                        env.set_goal_idx(goal_idx)

                        driver(eval_policy, episodes=1)
                        ep = driver._eps[0] # get episode data of 1st env.
                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                        # score = float(ep['reward'].astype(np.float64).sum())

                        all_metric_success.append(np.max(ep['is_rate_obs_success']))
                        all_env_original_success.append(np.max(ep['is_terminal']))

                        if ep['observation'][-1].shape[0] == ep['goal'][-1].shape[0]:
                            distance_from_goal = np.linalg.norm(ep['observation'][-1] - ep['goal'][-1])
                            all_distance_from_goal.append(distance_from_goal)

                            if config.task == "PegInsertionSide-v0":
                                distance_from_goal_object = np.linalg.norm(env.obs2goal(ep['observation'][-1]) - env.obs2goal(ep['goal'][-1]))
                                all_distance_from_goal_object.append(distance_from_goal_object)


                        if should_video:
                            """ rendering based on state."""
                            # render the goal img and rollout
                            ep_video = episode_render_fn(env, ep, if_eval=True)
                            
                            # imageio.mimsave(f'ep_{goal_idx}.gif', ep_video)
                            all_ep_video.append(ep_video[None])  # 1 x T x H x W x C



                # single_ep_video = all_ep_video[-1]
                # gif = np.squeeze(single_ep_video)
                # imageio.mimsave(f'single.gif', gif)
                all_ep_video = np.concatenate(all_ep_video, 3)
                
    
                # gif = np.squeeze(all_ep_video)
                # imageio.mimsave(f'ep_all.gif', gif)
                train_seed_all_ep_video = np.concatenate(train_seed_all_ep_video, 3)

                logger.video(f'eval_gc_policy', all_ep_video)
                logger.video(f'eval_gc_policy_train_seed', train_seed_all_ep_video)

                # collect all the goal success metrics and get average
                all_metric_success = np.mean(all_metric_success)
                all_env_original_success = np.mean(all_env_original_success)
                if len(all_distance_from_goal) > 0:
                    all_distance_from_goal = np.mean(all_distance_from_goal)
                else:
                    all_distance_from_goal = 0

                if len(all_distance_from_goal_object) > 0:
                    all_distance_from_goal_object = np.mean(all_distance_from_goal_object)
                else:
                    all_distance_from_goal_object = 0

                logger.scalar('mean_eval_goal_all_success', all_metric_success)
                logger.scalar('mean_eval_goal_all_env_original_success', all_env_original_success)
                logger.scalar('mean_eval_goal_all_distance_from_goal', all_distance_from_goal)
                logger.scalar('mean_eval_goal_all_distance_from_goal_object', all_distance_from_goal_object)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                train_seed_all_metric_success = np.mean(train_seed_all_metric_success)
                train_seed_all_env_original_success = np.mean(train_seed_all_env_original_success)
                if len(train_seed_all_distance_from_goal) > 0:
                    train_seed_all_distance_from_goal = np.mean(train_seed_all_distance_from_goal)
                else:
                    train_seed_all_distance_from_goal = 0

                if len(train_seed_all_distance_from_goal_object) > 0:
                    train_seed_all_distance_from_goal_object = np.mean(train_seed_all_distance_from_goal_object)
                else:
                    train_seed_all_distance_from_goal_object = 0

                logger.scalar('mean_eval_goal_train_seed_all_success', train_seed_all_metric_success)
                logger.scalar('mean_eval_goal_train_seed_all_env_original_success', train_seed_all_env_original_success)
                logger.scalar('mean_eval_goal_train_seed_all_distance_from_goal', train_seed_all_distance_from_goal)
                logger.scalar('mean_eval_goal_train_seed_all_distance_from_goal_object', train_seed_all_distance_from_goal_object)

                logger.write()


            
            def eval_fn_with_process_seed_reset(driver, eval_policy, logger, goal_optimizer=None):
                env = driver._envs[0]
                env = env._env
                env.if_eval_rate_goal = True
                eval_goal_num = len(env.train_seed_list)
                num_eval_eps = 2
                ep_demo_goal_rate_list = [0.3, 0.5, 0.8, 1]
                # ep_demo_goal_rate_list = [1]  # need to be deleted
                all_ep_video = []
                goals = []
                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                all_env_original_success = []
                all_distance_from_goal = []
                all_distance_from_goal_object = []
                ep_metrics = collections.defaultdict(list)

                rander_goal_list = [1, 3]
                # rander_goal_list = range(eval_goal_num)


                for goal_idx in range(eval_goal_num):

                    goal_video = []

                    for demo_goal_rate in ep_demo_goal_rate_list:

                        for ep_idx in range(num_eval_eps):

                            should_video = episode_render_fn is not None and goal_idx in rander_goal_list and ep_idx == 0

                            env.goal_rate = demo_goal_rate  
                            env.set_goal_idx(goal_idx)

                            driver(eval_policy, episodes=1, goal_optimizer=goal_optimizer)
                            ep = driver._eps[0] # get episode data of 1st env.
                            ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                            # score = float(ep['reward'].astype(np.float64).sum())

                            k = 'goal_' + str(goal_idx) + '_success' + '_rate_' + str(demo_goal_rate)

                            ep_metrics[k].append(np.max(ep['is_rate_obs_success']))
                            all_metric_success.append(np.max(ep['is_rate_obs_success']))

                            if ep['observation'][-1].shape[0] == ep['goal'][-1].shape[0]:
                                distance_from_goal = np.linalg.norm(ep['observation'][-1] - ep['goal'][-1])
                                all_distance_from_goal.append(distance_from_goal)

                                if config.task == "PegInsertionSide-v0" and demo_goal_rate == ep_demo_goal_rate_list[-1]:
                                    distance_from_goal_object = np.linalg.norm(env.obs2goal(ep['observation'][-1]) - env.obs2goal(ep['goal'][-1]))
                                    all_distance_from_goal_object.append(distance_from_goal_object)

                            k2 = 'goal_all_success' + '_rate_' + str(demo_goal_rate)

                            ep_metrics[k2].append(np.max(ep['is_rate_obs_success']))

                            if demo_goal_rate == 1:

                                all_env_original_success.append(np.max(ep['is_terminal']))

                            if should_video:
                                """ rendering based on state."""
                                # render the goal img and rollout
                                ep_video = episode_render_fn(env, ep, if_eval=True)
                                goal_video.append(ep_video[None])  # 1 x T x H x W x C


                    if goal_idx in rander_goal_list:
                        goal_video = np.concatenate(goal_video, 3) # ep_num x T x H x W x C
                        all_ep_video.append(goal_video)

                all_ep_video = np.concatenate(all_ep_video, 2)

                logger.video(f'eval_gc_policy', all_ep_video)

                # collect all the goal success metrics and get average
                all_metric_success = np.mean(all_metric_success)
                all_env_original_success = np.mean(all_env_original_success)
                if len(all_distance_from_goal) > 0:
                    all_distance_from_goal = np.mean(all_distance_from_goal)
                else:
                    all_distance_from_goal = 0

                if len(all_distance_from_goal_object) > 0:
                    all_distance_from_goal_object = np.mean(all_distance_from_goal_object)
                else:
                    all_distance_from_goal_object = 0

                logger.scalar('mean_eval_goal_all_success', all_metric_success)
                logger.scalar('mean_eval_goal_all_env_original_success', all_env_original_success)
                logger.scalar('mean_eval_goal_all_distance_from_goal', all_distance_from_goal)
                logger.scalar('mean_eval_goal_all_distance_from_goal_object', all_distance_from_goal_object)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()


            
            def eval_fn_goal_predictor(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env

                if env.goal_predictor is None:
                    print("Goal predictor is not available for eval function: eval_fun_goal_predictor.")
                    return
                    
                env.if_eval_random_seed_with_gp = True
                num_eval_eps = 50

                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    should_video = episode_render_fn is not None and ep_idx < 5

                    driver(eval_policy, episodes=1)
                    ep = driver._eps[0] # get episode data of 1st env.
                    ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}


                    if should_video:
                        ep_video = episode_render_fn(env, ep, if_eval=True)
                        all_ep_video.append(ep_video[None])  # 1 x T x H x W x C
                                            
                    all_metric_success.append(np.max(ep['is_terminal']))

                    # ep_metrics[k].append(np.max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0) # ep_num x T x H x W x C

                logger.video(f'eval_gc_policy_goal_predictor', all_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success_goal_predictor', all_metric_success)

                # for key, value in ep_metrics.items():

                #     logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

                env.if_eval_random_seed_with_gp = False


            def mix_eval_fn(driver, eval_policy, logger):

                print("Begin to evaluate with demo final obs as goal.")
                eval_fn_with_demo_seed_reset(driver, eval_policy, logger)

                print("Begin to evaluate with goal predictor.")
                eval_fn_goal_predictor(driver, eval_policy, logger)

            if config.if_use_demo:

                # return eval_fn_with_seed_reset
                return mix_eval_fn


        elif config.task == "PickAndPlace" or config.task == "FetchSlide-v2" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:
    
            def eval_fn_with_random_reset(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                num_eval_eps = 50

                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    should_video = ep_idx < 5

                    obs = env.reset()

                    ep_goal = obs['goal']

                    for k in obs:
                        obs[k] = np.stack([obs[k]])
                    
                    state = None
                    done = False

                    ep = []
                    ep_video = []

                    while not done:

                        # print(obs)
                        actions, state = eval_policy(obs, state)

                        actions = [{k: np.array(actions[k][0]) for k in actions}]

                        obs = env.step(actions[0])

                        for k in obs:
                            obs[k] = np.stack([obs[k]])

                        done = obs['is_last']

                        ep.append(obs)

                        if should_video:
                            ep_video.append(env.render())

                    if should_video:
                        ep_video = np.stack(ep_video, 0)  
                        all_ep_video.append(ep_video[None])  # 1 x T x H x W x C

                    ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}

                    if 0.42 <= ep_goal[2] < 0.52:
                        k = 'goal_low_success'

                    elif 0.52 <= ep_goal[2] < 0.62:
                        k = 'goal_medium_success'

                    elif 0.62 <= ep_goal[2]:
                        k = 'goal_high_success'

                    ep_metrics[k].append(np.max(ep['is_terminal']))


                    all_metric_success.append(np.max(ep['is_terminal']))


                all_ep_video = np.concatenate(all_ep_video, 0) # ep_num x T x H x W x C

                logger.video(f'eval_gc_policy', all_ep_video)


                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))


                logger.write()
            
            def eval_fn_with_seed_reset(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                eval_goal_num = 10
                num_eval_eps = 5
                executions = []
                goals = []
                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for goal in range(eval_goal_num):

                    for ep_idx in range(num_eval_eps):

                        should_video = ep_idx == 0

                        env.set_goal_idx(goal)

                        obs = env.reset()

                        for k in obs:
                            obs[k] = np.stack([obs[k]])
                        
                        state = None
                        done = False

                        ep = []
                        ep_video = []

                        while not done:

                            # print(obs)
                            actions, state = eval_policy(obs, state)

                            actions = [{k: np.array(actions[k][0]) for k in actions}]

                            obs = env.step(actions[0])

                            for k in obs:
                                obs[k] = np.stack([obs[k]])

                            done = obs['is_last']

                            ep.append(obs)

                            if should_video:
                                ep_video.append(env.render())

                        if should_video:
                            ep_video = np.stack(ep_video, 0)  
                            all_ep_video.append(ep_video[None])  # 1 x T x H x W x C

                        ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                        k = 'goal_' + str(goal) + '_success'
                        ep_metrics[k].append(np.max(ep['is_terminal']))

                        all_metric_success.append(np.max(ep['is_terminal']))


                all_ep_video = np.concatenate(all_ep_video, 0) # ep_num x T x H x W x C

                logger.video(f'eval_gc_policy', all_ep_video)


                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()
            
            
            def eval_fn_with_process_seed_reset(driver, eval_policy, logger):
                env = driver._envs[0]
                env = env._env
                env.if_eval_rate_goal = True
                eval_goal_num = len(env.seed_list)
                num_eval_eps = 5
                ep_demo_goal_rate_list = [0.5, 0.7, 1]
                # ep_demo_goal_rate_list = [1]
                all_ep_video = []
                goals = []
                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                ep_metrics = collections.defaultdict(list)

                rander_goal_list = [2, 8]

                for goal_idx in range(eval_goal_num):

                    goal_video = []

                    for demo_goal_rate in ep_demo_goal_rate_list:

                        for ep_idx in range(num_eval_eps):

                            should_video = episode_render_fn is not None and goal_idx in rander_goal_list and ep_idx == 0

                            env.goal_rate = demo_goal_rate  
                            env.set_goal_idx(goal_idx)

                            driver(eval_policy, episodes=1)
                            ep = driver._eps[0] # get episode data of 1st env.
                            ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
                            # score = float(ep['reward'].astype(np.float64).sum())

                            k = 'goal_' + str(goal_idx) + '_success' + '_rate_' + str(demo_goal_rate)

                            ep_metrics[k].append(np.max(ep['is_terminal']))
                            all_metric_success.append(np.max(ep['is_terminal']))

                            k2 = 'goal_all_success' + '_rate_' + str(demo_goal_rate)

                            ep_metrics[k2].append(np.max(ep['is_terminal']))

                            if should_video:
                                """ rendering based on state."""
                                # render the goal img and rollout
                                ep_video = episode_render_fn(env, ep)
                                goal_video.append(ep_video[None])  # 1 x T x H x W x C


                    if goal_idx in rander_goal_list:
                        goal_video = np.concatenate(goal_video, 3) # ep_num x T x H x W x C
                        all_ep_video.append(goal_video)

                all_ep_video = np.concatenate(all_ep_video, 2)

                logger.video(f'eval_gc_policy', all_ep_video)

                # collect all the goal success metrics and get average
                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

                


            
            def eval_fn_with_gp(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                env.if_eval_random_seed_with_gp = True
                num_eval_eps = 50

                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                # all_ep_video = []
                success_ep_video = []
                fail_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    # should_video = episode_render_fn is not None and ep_idx < 5

                    driver(eval_policy, episodes=1)
                    ep = driver._eps[0] # get episode data of 1st env.
                    ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}

                    # if should_video:
                    #     ep_video = episode_render_fn(env, ep)
                    #     all_ep_video.append(ep_video[None])  # 1 x T x H x W x C
                    
                    is_success = np.max(ep['is_terminal'])
                    all_metric_success.append(is_success)

                    if is_success and len(success_ep_video) < 10:
                        ep_video = episode_render_fn(env, ep)
                        success_ep_video.append(ep_video[None])

                    if not is_success and len(fail_ep_video) < 10:
                        ep_video = episode_render_fn(env, ep)
                        fail_ep_video.append(ep_video[None])  # 1 x T x H x W x C


                env.if_eval_random_seed_with_gp = False

                if len(success_ep_video) > 0:
                    success_ep_video = np.concatenate(success_ep_video, 0) # ep_num x T x H x W x C
                    logger.video(f'eval_gc_policy_random_seed_using_gp_success', success_ep_video)

                if len(fail_ep_video) > 0:
                    fail_ep_video = np.concatenate(fail_ep_video, 0) # ep_num x T x H x W x C
                    logger.video(f'eval_gc_policy_random_seed_using_gp_fail', fail_ep_video)

                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success_gp', all_metric_success)

                for key, value in ep_metrics.items():
                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()


            def eval_fn(driver, eval_policy, logger):

                env = driver._envs[0]

                env = env._env
                num_eval_eps = 50

                # key is metric name, value is list of size num_eval_eps
                all_metric_success = []
                all_ep_video = []
                ep_metrics = collections.defaultdict(list)

                for ep_idx in range(num_eval_eps):

                    should_video = episode_render_fn is not None and ep_idx < 5

                    driver(eval_policy, episodes=1)
                    ep = driver._eps[0] # get episode data of 1st env.
                    ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}


                    if should_video:
                        ep_video = episode_render_fn(env, ep)
                        all_ep_video.append(ep_video[None])  # 1 x T x H x W x C
                                            
                    all_metric_success.append(np.max(ep['is_terminal']))

                    if config.task == "PickAndPlace":
                        ep_goal = ep['goal'][0]

                        if 0.42 <= ep_goal[2] < 0.52:
                            k = 'goal_low_success'

                        elif 0.52 <= ep_goal[2] < 0.62:
                            k = 'goal_medium_success'

                        elif 0.62 <= ep_goal[2]:
                            k = 'goal_high_success'

                        ep_metrics[k].append(np.max(ep['is_terminal']))

                all_ep_video = np.concatenate(all_ep_video, 0) # ep_num x T x H x W x C

                logger.video(f'eval_gc_policy', all_ep_video)


                all_metric_success = np.mean(all_metric_success)
                logger.scalar('mean_eval_goal_all_success', all_metric_success)

                for key, value in ep_metrics.items():

                    logger.scalar(f'mean_eval_{key}', np.mean(value))

                logger.write()

            if config.if_use_demo:
                return eval_fn_with_process_seed_reset
        
        
        else:
            raise NotImplementedError

        return eval_fn


    
    def make_obs2goal_fn(self, config):
        obs2goal = None
        if "demofetchpnp" in config.task:
            def obs2goal(obs):
                return obs
            
        elif config.task == "PickAndPlace":
            def obs2goal(obs):
                return obs[..., 3:6]
            
        elif config.task == "FetchPush-v2" or config.task == "FetchSlide-v2":
            def obs2goal(obs):
                return obs[..., 3:6]
            
        elif "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task:
                
            def obs2goal(obs):

                if "HandManipulateBlockRotateXYZ" in config.task or "HandManipulatePenRotate" in config.task or "HandManipulateEggRotate" in config.task:
                    return obs[..., -4:]
                else:
                    return obs[..., -7:]
            
        elif config.task == "StackCube-v0":

            def obs2goal(obs):
                return obs[..., -9:]
            # def obs2goal(obs):
            #     return obs

        elif config.task == "PegInsertionSide-v0":

            def obs2goal(obs):
                return obs[..., -9:]
            
        elif config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1":

            def obs2goal(obs):
                return obs
            
        return obs2goal


    
    def make_space_explored_plot_fn(self, config):

        space_explored_plot_fn = None 

        if 'demofetchpnp' in config.task:
            def space_explored_plot_fn(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50):
                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                all_observations = []
                value_list = []
                all_goals = []

                def unnorm_ob(ob):
                    return env.obs_min + ob * (env.obs_max -    env.obs_min)

                for ep_count, episode in enumerate(episodes):
                    # 2. Adding episodes to the batch
                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)
                    # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        all_observations.append(tf.stack(chunk[obs_key]))
                        all_goals.append(tf.stack(chunk[goal_key]))

                all_observations = np.concatenate(all_observations)
                all_observations = all_observations.reshape(-1, all_observations.shape[-1])
                all_observations = unnorm_ob(all_observations)
                all_obj_obs_pos = np.split(all_observations[:, 5:5+3*env.n], env.n, axis=1)

                all_goals = np.concatenate(all_goals)[:, 0]
                all_goals = unnorm_ob(all_goals)
                all_obj_g_pos = np.split(all_goals[:, 5:5+3*env.n], env.n, axis=1)

                plot_dims = [[1, 2]]
                plot_dim_name = dict([(0,'x'), (1,'y'), (2,'z')])
                def plot_axes(axes, data, cmap, title, zorder):
                    for ax, pd in zip(axes, plot_dims):
                        ax.scatter(x=data[:, pd[0]],
                            y=data[:, pd[1]],
                            s=1,
                            c=np.arange(len(data)),
                            cmap=cmap,
                            zorder=zorder,
                        )
                        ax.set_title(f"{title} {plot_dim_name[pd[0]]}{plot_dim_name[pd[1]]}", fontdict={'fontsize':10})

                fig, all_axes = plt.subplots(1,2+env.n, figsize=(1+(2+env.n*3),2))

                g_ax = all_axes[0]
                p2evalue_ax = all_axes[-1]
                obj_axes = all_axes[1:-1]
                obj_colors = ['Reds', 'Blues', 'Greens']
                for obj_ax, obj_pos, obj_g_pos, obj_color in zip(obj_axes, all_obj_obs_pos, all_obj_g_pos, obj_colors):
                    plot_axes([obj_ax], obj_pos, obj_color, f"State ", 3)
                    plot_axes([g_ax], obj_g_pos, obj_color, f"Goal ", 3)

                # Plot temporal distance reward.
                # x_min, x_max = 1.2, 1.65
                # y_min, y_max = 0.3, 0.7
                # x_div = y_div = 100
                # x = np.linspace(x_min, x_max, x_div)
                # y = np.linspace(y_min, y_max, y_div)
                # XZ = X, Z = np.meshgrid(x, y)
                # XZ = np.stack([X, Z], axis=-1)
                # XZ = XZ.reshape(x_div * y_div, 2)

                # start_pos = np.array([1.3, 0.65, 0.41])
                # goal_pos = start_pos + np.array([0, 0, 0.4])
                # goal_vec = np.zeros((x_div*y_div, 3))
                # goal_vec[:,0] = goal_pos[0]
                # goal_vec[:,1] = goal_pos[1]
                # goal_vec[:,2] = goal_pos[2]

                # observation = np.zeros((x_div*y_div, 3))
                # observation[:, 0] = XZ[:, 0]
                # observation[:, 1] = goal_pos[1]
                # observation[:, 2] = XZ[:, 1]

                # obs = {"observation": observation, "goal": goal_vec, "reward": np.zeros(len(XZ)), "discount": np.ones(len(XZ)), "is_terminal": np.zeros(len(XZ))}
                # temporal_dist = agnt.temporal_dist(obs)
                # if config.gc_reward == 'dynamical_distance':
                #     im = rew_ax.tricontourf(XZ[:, 0], XZ[:, 1], temporal_dist, zorder=1)
                #     rew_ax.scatter(x=[goal_pos[0]], y=[goal_pos[2]], c="r", marker="*", s=20, zorder=2)
                #     rew_ax.scatter(x=[start_pos[0]], y=[start_pos[2]], c="b", marker=".", s=20, zorder=2)
                #     plt.colorbar(im, ax=rew_ax)
                limits = [[0.5, 1.0], [0.4, 0.6]] if 'walls' in config.task else [[1, 1.6], [0.3, 0.7]]
                for _ax in all_axes:
                    _ax.set_xlim(limits[0])
                    _ax.set_ylim(limits[1])
                    _ax.axes.get_yaxis().set_visible(False)

                # plot p2e value function
                values = tf.concat(value_list, axis = 0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                for obj_ax, obj_pos, obj_color in zip(obj_axes, all_obj_obs_pos, obj_colors):
                    p2e_scatter = p2evalue_ax.scatter(
                        x=obj_pos[:,plot_dims[0][0]],
                        y=obj_pos[:,plot_dims[0][1]],
                        s=1,
                        c=values,
                        cmap=cm,
                        zorder=3,
                    )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis = 0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

        elif config.task in {'umazefulldownscale','a1umazefulldownscale', 'hardumazefulldownscale'}:
            def space_explored_plot_fn(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50):
            # 1. Load all episodes
                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                obs = []
                value_list = []
                goals = []
                for ep_count, episode in enumerate(episodes):
                    # 2. Adding episodes to the batch
                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)
                    # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))
            # 4. Plotting
                fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(1, 3, figsize=(1, 3))
                xlim = np.array([-1, 5.25])
                ylim = np.array([-1, 5.25])
                if config.task == 'a1umazefulldownscale':
                    xlim /= 2.0
                    ylim /= 2.0
                elif config.task == 'hardumazefulldownscale':
                    xlim = np.array([-1, 5.25])
                    ylim = np.array([-1, 9.25])

                state_ax.set(xlim=xlim, ylim=ylim)
                p2evalue_ax.set(xlim=xlim, ylim=ylim)
                goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
                obs_list = tf.concat(obs, axis = 0)
                before = obs_list[:,:goal_time_limit,:]
                before = before[:,:,:2]
                ep_order_before = tf.range(before.shape[0])[:, None]
                ep_order_before = tf.repeat(ep_order_before, before.shape[1], axis=1)
                before = tf.reshape(before, [before.shape[0]*before.shape[1], 2])
                after = obs_list[:,goal_time_limit:,:]
                after = after[:,:,:2]
                ep_order_after = tf.range(after.shape[0])[:, None]
                ep_order_after = tf.repeat(ep_order_after, after.shape[1], axis=1)
                after = tf.reshape(after, [after.shape[0]*after.shape[1], 2])
                # obs_list = tf.concat(obs, axis = 0)
                # obs_list = obs_list[:,:,:2]
                # ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
                # ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #    Num_ep x T
                # obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                # ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
                ep_order_before = tf.reshape(ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
                ep_order_after = tf.reshape(ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
                goal_list = tf.concat(goals, axis = 0)[:, 0, :2]
                goal_list = tf.reshape(goal_list, [-1, 2])
                # plt.scatter(
                #         x=obs_list[:,0],
                #         y=obs_list[:,1],
                #         s=1,
                #         c=ep_order,
                #         cmap='Blues',
                #         zorder=3,
                #         )
                state_ax.scatter(
                        x=before[:,0],
                        y=before[:,1],
                        s=1,
                        c=ep_order_before,
                        cmap='Blues',
                        zorder=3,
                        )
                state_ax.scatter(
                        x=after[:,0],
                        y=after[:,1],
                        s=1,
                        c=ep_order_after,
                        cmap='Greens',
                        zorder=3,
                        )
                state_ax.scatter(
                        x=goal_list[:,0],
                        y=goal_list[:,1],
                        s=1,
                        c=np.arange(goal_list.shape[0]),
                        cmap='Reds',
                        zorder=3,
                        )
                x_min, x_max = xlim[0], xlim[1]
                y_min, y_max = ylim[0], ylim[1]
                x_div = y_div = 100
                if config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
                    other_dims = np.concatenate([[6.08193526e-01,    9.87496030e-01,
                    1.82685311e-03, -6.82827458e-03,    1.57485326e-01,    5.14617396e-02,
                    1.22386603e+00, -6.58701813e-02, -1.06980319e+00,    5.09069276e-01,
                    -1.15506861e+00,    5.25953435e-01,    7.11716520e-01], np.zeros(14)])
                elif config.task == 'a1umazefulldownscale':
                    other_dims = np.concatenate([[0.24556014,    0.986648,        0.09023235, -0.09100603,
                        0.10050705, -0.07250207, -0.01489305,    0.09989551, -0.05246516, -0.05311238,
                        -0.01864055, -0.05934234,    0.03910208, -0.08356607,    0.05515265, -0.00453086,
                        -0.01196933], np.zeros(18)])
                gx = 0.0
                gy = 4.2
                if config.task == 'a1umazefulldownscale':
                    gx /= 2
                    gy /= 2
                elif config.task == 'hardumazefulldownscale':
                    gx = 4.2
                    gy = 8.2

                x = np.linspace(x_min, x_max, x_div)
                y = np.linspace(y_min, y_max, y_div)
                XY = X, Y = np.meshgrid(x, y)
                XY = np.stack([X, Y], axis=-1)
                XY = XY.reshape(x_div * y_div, 2)
                XY_plus = np.hstack((XY, np.tile(other_dims, (XY.shape[0], 1))))
                goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
                goal_vec[:,0] = goal_vec[:,0] + gx
                goal_vec[:,1] = goal_vec[:,1] + gy
                goal_vec[:,2:] = goal_vec[:,2:] + other_dims
                obs = {"observation": XY_plus, "goal": goal_vec, "reward": np.zeros(XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
                temporal_dist = agnt.temporal_dist(obs)
                if config.gc_reward == 'dynamical_distance':
                    td_plot = dd_ax.tricontourf(XY[:, 0], XY[:, 1], temporal_dist)
                    dd_ax.scatter(x = obs['goal'][0][0], y = obs['goal'][0][1], c="r", marker="*", s=20, zorder=2)
                    dd_ax.scatter(x = before[0][0], y = before[0][1], c="b", marker=".", s=20, zorder=2)
                    plt.colorbar(td_plot, ax=dd_ax)
                    dd_ax.set_title('temporal distance')

                obs_list = obs_list[:, :, :2]
                obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                values = tf.concat(value_list, axis = 0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                p2e_scatter = p2evalue_ax.scatter(
                    x=obs_list[:,0],
                    y=obs_list[:,1],
                    s=1,
                    c=values,
                    cmap=cm,
                    zorder=3,
                )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig = plt.gcf()
                fig.set_size_inches(10, 3)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis = 0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()
        
        elif 'pointmaze' in config.task:
            def space_explored_plot_fn(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 100):
            # 1. Load all episodes
                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                obs = []
                goals = []
                reward_list = []
                for ep_count, episode in enumerate(episodes[::ep_subsample]):
                    # 2. Adding episodes to the batch
                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)
                    # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes[::ep_subsample]) - 1):
                        end = ep_count
                        # chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))
            # 4. Plotting
                fig, ax = plt.subplots(1, 1, figsize=(1, 1))
                ax.set(xlim=(-1, 11), ylim=(-1, 11))
                maze.maze.plot(ax) # plot the walls
                obs_list = tf.concat(obs, axis = 0)
                ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
                ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #    Num_ep x T
                obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
                goal_list = tf.concat(goals, axis = 0)[:, 0, :]
                goal_list = tf.reshape(goal_list, [-1, 2])
                plt.scatter(
                        x=obs_list[:,0],
                        y=obs_list[:,1],
                        s=1,
                        c=ep_order,
                        cmap='Blues',
                        zorder=3,
                        )
                plt.scatter(
                        x=goal_list[:,0],
                        y=goal_list[:,1],
                        s=1,
                        c=np.arange(goal_list.shape[0]),
                        cmap='Reds',
                        zorder=3,
                        )
                fig = plt.gcf()
                plt.title('states')
                fig.set_size_inches(8, 8)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis = 0)
                logger.image('state_occupancy', image_from_plot)

        elif 'dmc_walker_walk_proprio' == config.task:
            def space_explored_plot_fn(eval_env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50):
            # 1. Load all episodes
                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]  
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                obs = []
                value_list = []
                goals = []
                for ep_count, episode in enumerate(episodes):
                    # 2. Adding episodes to the batch
                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)
                    # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)  

                        obs.append(tf.stack(chunk[obs_key]))  # obs
                        goals.append(tf.stack(chunk[goal_key]))  # goal
            # 4. Plotting
                        
                
                fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(1, 3, figsize=(1, 3))

                xlim = np.array([-20.0, 20.0])
                ylim = np.array([-1.3, 1.0])

                state_ax.set(xlim=xlim, ylim=ylim)
                p2evalue_ax.set(xlim=xlim, ylim=ylim)

                
                
                goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
                obs_list = tf.concat(obs, axis = 0)

                # before steps
                before = obs_list[:,:goal_time_limit,:]
                before = before[:,:,:2]

                
                ep_order_before = tf.range(before.shape[0])[:, None]
                ep_order_before = tf.repeat(ep_order_before, before.shape[1], axis=1)
                before = tf.reshape(before, [before.shape[0]*before.shape[1], 2])

                # after steps
                after = obs_list[:,goal_time_limit:,:]
                after = after[:,:,:2]

                
                ep_order_after = tf.range(after.shape[0])[:, None]
                ep_order_after = tf.repeat(ep_order_after, after.shape[1], axis=1)
                after = tf.reshape(after, [after.shape[0]*after.shape[1], 2])
                # obs_list = tf.concat(obs, axis = 0)
                # obs_list = obs_list[:,:,:2]
                # ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
                # ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #    Num_ep x T
                # obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                # ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
                ep_order_before = tf.reshape(ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
                ep_order_after = tf.reshape(ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
                goal_list = tf.concat(goals, axis = 0)[:, 0, :2]
                goal_list = tf.reshape(goal_list, [-1, 2])
                # plt.scatter(
                #         x=obs_list[:,0],
                #         y=obs_list[:,1],
                #         s=1,
                #         c=ep_order,
                #         cmap='Blues',
                #         zorder=3,
                #         )
                state_ax.scatter(
                        y=before[:,0],
                        x=before[:,1],
                        s=1,
                        c=ep_order_before,
                        cmap='Blues',
                        zorder=3,
                        )
                state_ax.scatter(
                        y=after[:,0],
                        x=after[:,1],
                        s=1,
                        c=ep_order_after,
                        cmap='Greens',
                        zorder=3,
                        )
                state_ax.scatter(
                        y=goal_list[:,0],
                        x=goal_list[:,1],
                        s=1,
                        c=np.arange(goal_list.shape[0]),
                        cmap='Reds',
                        zorder=3,
                        )
                
                state_ax.set_title('space explored')


                
                x_min, x_max = xlim[0], xlim[1]
                y_min, y_max = ylim[0], ylim[1]
                x_div = y_div = 100
                other_dims = np.array([ 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1])
                gx = 5.0
                gy = 0.0

                x = np.linspace(x_min, x_max, x_div)
                y = np.linspace(y_min, y_max, y_div)
                XY = X, Y = np.meshgrid(y, x)
                XY = np.stack([X, Y], axis=-1)
                XY = XY.reshape(x_div * y_div, 2)
                XY_plus = np.hstack((XY, np.tile(other_dims, (XY.shape[0], 1))))
                # swap first and second element
                # import ipdb; ipdb.set_trace()


                goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
                goal_vec[:,0] = goal_vec[:,0] + gy
                goal_vec[:,1] = goal_vec[:,1] + gx
                goal_vec[:,2:] = goal_vec[:,2:] + other_dims

                obs = {"qpos": XY_plus, "goal": goal_vec, "reward": np.zeros(XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
                temporal_dist = agnt.temporal_dist(obs)
                if config.gc_reward == 'dynamical_distance':
                    td_plot = dd_ax.tricontourf(XY[:, 1], XY[:, 0], temporal_dist)
                    dd_ax.scatter(y = obs['goal'][0][0], x = obs['goal'][0][1], c="r", marker="*", s=20, zorder=2)
                    dd_ax.scatter(y = before[0][0], x = before[0][1], c="b", marker=".", s=20, zorder=2)
                    plt.colorbar(td_plot, ax=dd_ax)
                    dd_ax.set_title('temporal distance')


                
                obs_list = obs_list[:, :, :2]
                obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                values = tf.concat(value_list, axis = 0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                p2e_scatter = p2evalue_ax.scatter(
                    y=obs_list[:,0],
                    x=obs_list[:,1],
                    s=1,
                    c=values,
                    cmap=cm,
                    zorder=3,
                )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig = plt.gcf()
                fig.set_size_inches(12, 3)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis = 0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

                

                # def get_centroids():

                #     try:
                #         centroids = agnt.wm.cluster.centroids()
                #     except:
                #         return None

                #     # print(centroids)
                #     # print(centroids.shape)

                #     centroids = tf.convert_to_tensor(centroids.numpy(), dtype=agnt.wm.dtype)

                #     # centroid_1_embed = centroids[0]

                #     state = None
                #     if state is None:
                #         latent = agnt.wm.rssm.initial(self.config.cluster['n_latent_landmarks'])
                #         action = tf.zeros((self.config.cluster['n_latent_landmarks'],) + agnt.act_space.shape)
                #         state = latent, action

                #     latent, action = state

                #     # print("latent", latent)
                #     # print(centroids)

                #     latent, _ = agnt.wm.rssm.obs_step(latent, action, centroids, True, True)

                #     decoder = agnt.wm.heads['decoder']

                #     feat = agnt.wm.rssm.get_feat(latent)

                #     # print("feat", feat)

                #     centroids_decoded_dist = decoder(feat)

                #     centroids_decoded = centroids_decoded_dist[agnt.state_key].mean()


                #     return centroids_decoded
                
                # centroids_decoded = get_centroids()

                # if centroids_decoded is None:

                #     return

                # def render_fn(env, ep):

                #     all_img = []
                #     inner_env = env._env._env._env._env
                #     for qpos in [*ep]:
                #         size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
                #         inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
                #         inner_env.physics.step()
                #         img = env.render()
                #         all_img.append(img)

                #     ep_img = np.stack(all_img[1:], 0)
                #     return ep_img

                # # print("centroids_decoded", centroids_decoded)
                # centroids_decoded_images = render_fn(eval_env, centroids_decoded)

                # centroids_decoded_images = np.array(centroids_decoded_images)

                # # print(centroids_decoded_images.shape)
                # centroids_stacked_image = np.concatenate(centroids_decoded_images, axis= 1 )
                # centroids_stacked_image = np.expand_dims(centroids_stacked_image, axis = 0)

                # logger.image('Cluster centroids from ' + self.config.centroids_assign_space, centroids_stacked_image)

                # # print(centroids_decoded_images)
                # for i in range(len(centroids_decoded_images)):
                #     image = centroids_decoded_images[i]
                #     imageio.imwrite(save_path + str(i) + '_centroid.png', image)

        elif 'dmc_humanoid_walk_proprio' == config.task:
            def space_explored_plot_fn(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50):
            # 1. Load all episodes
                wm = agnt.wm
                obs_key = agnt.state_key
                goal_key = agnt.goal_key
                episodes = list(complete_episodes.values())
                num_goals = min(int(config.eval_every), len(episodes))
                recent_episodes = episodes[-num_goals:]
                if len(episodes) > num_goals:
                    old_episodes = episodes[:-num_goals][::5]
                    old_episodes.extend(recent_episodes)
                    episodes = old_episodes
                else:
                    episodes = recent_episodes

                obs = []
                value_list = []
                goals = []
                for ep_count, episode in enumerate(episodes):
                    # 2. Adding episodes to the batch
                    if (ep_count % batch_size) == 0:
                        start = ep_count
                        chunk = collections.defaultdict(list)
                    sequence = {
                        k: convert(v[::step_subsample])
                        for k, v in episode.items() if not k.startswith('log_')}
                    data = wm.preprocess(sequence)
                    for key, value in data.items():
                        chunk[key].append(value)
                    # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
                    if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
                        end = ep_count
                        chunk = {k: tf.stack(v) for k, v in chunk.items()}
                        embed = wm.encoder(chunk)
                        post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
                        chunk['feat'] = wm.rssm.get_feat(post)
                        value_fn = agnt._expl_behavior.ac._target_critic
                        value = value_fn(chunk['feat']).mode()
                        value_list.append(value)

                        obs.append(tf.stack(chunk[obs_key]))
                        goals.append(tf.stack(chunk[goal_key]))

            # 4. Plotting
                fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(1, 3, figsize=(1, 3))
                xlim = np.array([-0.2, 1.2])
                ylim = np.array([-0.2, 1.2])

                state_ax.set(xlim=xlim, ylim=ylim)
                p2evalue_ax.set(xlim=xlim, ylim=ylim)
                goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
                obs_list = tf.concat(obs, axis = 0)
                before = obs_list[:,:goal_time_limit,:]
                before = before[:,:,:28]
                ep_order_before = tf.range(before.shape[0])[:, None]
                ep_order_before = tf.repeat(ep_order_before, before.shape[1], axis=1)
                before = tf.reshape(before, [before.shape[0]*before.shape[1], 28])
                after = obs_list[:,goal_time_limit:,:]
                after = after[:,:,:28]
                ep_order_after = tf.range(after.shape[0])[:, None]
                ep_order_after = tf.repeat(ep_order_after, after.shape[1], axis=1)
                after = tf.reshape(after, [after.shape[0]*after.shape[1], 28])
                # obs_list = tf.concat(obs, axis = 0)
                # obs_list = obs_list[:,:,:2]
                # ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
                # ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #    Num_ep x T
                # obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
                # ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
                ep_order_before = tf.reshape(ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
                ep_order_after = tf.reshape(ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
                goal_list = tf.concat(goals, axis = 0)[:, 0, :28]
                goal_list = tf.reshape(goal_list, [-1, 28])
                # plt.scatter(
                #         x=obs_list[:,0],
                #         y=obs_list[:,1],
                #         s=1,
                #         c=ep_order,
                #         cmap='Blues',
                #         zorder=3,
                #         )
                state_ax.scatter(
                        y=before[:,0],
                        x=before[:,1],
                        s=1,
                        c=ep_order_before,
                        cmap='Blues',
                        zorder=3,
                        )
                state_ax.scatter(
                        y=after[:,0],
                        x=after[:,1],
                        s=1,
                        c=ep_order_after,
                        cmap='Greens',
                        zorder=3,
                        )
                state_ax.scatter(
                        y=goal_list[:,0],
                        x=goal_list[:,1],
                        s=1,
                        c=np.arange(goal_list.shape[0]),
                        cmap='Reds',
                        zorder=3,
                        )

                # x_min, x_max = xlim[0], xlim[1]
                # y_min, y_max = ylim[0], ylim[1]
                # x_div = y_div = 100
                # other_dims = np.array([ 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1])
                # gx = 5.0
                # gy = 0.0

                # x = np.linspace(x_min, x_max, x_div)
                # y = np.linspace(y_min, y_max, y_div)
                # XY = X, Y = np.meshgrid(y, x)
                # XY = np.stack([X, Y], axis=-1)
                # XY = XY.reshape(x_div * y_div, 2)
                # XY_plus = np.hstack((XY, np.tile(other_dims, (XY.shape[0], 1))))
                # swap first and second element
                # import ipdb; ipdb.set_trace()


                # goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
                # goal_vec[:,0] = goal_vec[:,0] + gy
                # goal_vec[:,1] = goal_vec[:,1] + gx
                # goal_vec[:,2:] = goal_vec[:,2:] + other_dims

                # obs = {"qpos": XY_plus, "goal": goal_vec, "reward": np.zeros(XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
                # temporal_dist = agnt.temporal_dist(obs)
                # if config.gc_reward == 'dynamical_distance':
                #     td_plot = dd_ax.tricontourf(XY[:, 1], XY[:, 0], temporal_dist)
                #     dd_ax.scatter(y = obs['goal'][0][0], x = obs['goal'][0][1], c="r", marker="*", s=20, zorder=2)
                #     dd_ax.scatter(y = before[0][0], x = before[0][1], c="b", marker=".", s=20, zorder=2)
                #     plt.colorbar(td_plot, ax=dd_ax)
                #     dd_ax.set_title('temporal distance')

                obs_list = obs_list[:, :, :28]
                obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 28])
                values = tf.concat(value_list, axis = 0)
                values = values.numpy().flatten()
                cm = plt.cm.get_cmap("viridis")
                p2e_scatter = p2evalue_ax.scatter(
                    y=obs_list[:,0],
                    x=obs_list[:,1],
                    s=1,
                    c=values,
                    cmap=cm,
                    zorder=3,
                )
                plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                p2evalue_ax.set_title('p2e value')

                fig = plt.gcf()
                fig.set_size_inches(10, 3)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis = 0)
                logger.image('state_occupancy', image_from_plot)
                plt.cla()
                plt.clf()

    
        return None


    def make_cem_vis_fn(self, config):

        vis_fn = None
        if config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
            num_vis = 10
            def vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger):
                elite_seq = tf.nest.map_structure(lambda x: tf.gather(x, elite_inds[:num_vis], axis=1), seq)
                elite_obs = wm.heads['decoder'](wm.rssm.get_feat(elite_seq))['observation'].mode()
                goal_states = tf.repeat(elite_samples[None, :num_vis], elite_obs.shape[0], axis=0).numpy() # T x topk x 2
                goal_list = goal_states[...,:2]
                goal_list = tf.reshape(goal_list, [-1, 2])

                fig, p2evalue_ax = plt.subplots(1, 1, figsize=(1, 3))
                p2evalue_ax.scatter(
                    x=goal_list[:,0],
                    y=goal_list[:,1],
                    s=1,
                    c='r',
                    zorder=5,
                )
                elite_obs = tf.transpose(elite_obs, (1,0,2))
                # (num_vis,horizon,29)
                first_half = elite_obs[:, :-10]
                first_half = first_half[:, ::10]
                second_half = elite_obs[:, -10:]
                traj = tf.concat([first_half, second_half], axis=1)
                p2evalue_ax.plot(
                        traj[:,:,0],
                        traj[:,:,1],
                        c='b',
                        zorder=4,
                        marker='.'
                )

                # plt.colorbar(p2e_scatter, ax=p2evalue_ax)
                if 'hard' in config.task:
                    p2evalue_ax.set(xlim=(-1, 5.25), ylim=(-1, 9.25))
                else:
                    p2evalue_ax.set(xlim=(-1, 5.25), ylim=(-1, 5.25))
                p2evalue_ax.set_title('elite goals and states')
                fig = plt.gcf()
                fig.set_size_inches(7, 6)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image_from_plot = np.expand_dims(image_from_plot, axis = 0)
                logger.image(f'top_{num_vis}_cem', image_from_plot)
                logger.write()

        elif 'pointmaze' in config.task:
            num_vis = 10
            def vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger):
                elite_seq = tf.nest.map_structure(lambda x: tf.gather(x, elite_inds[:num_vis], axis=1), seq)
                elite_obs = wm.heads['decoder'](wm.rssm.get_feat(elite_seq))['observation'].mode()

                goal_states = tf.repeat(elite_samples[None, :num_vis], elite_obs.shape[0], axis=0).numpy() # T x topk x 2
                goal_states = goal_states.reshape(-1,2)
                maze_states =    elite_obs.numpy().reshape(-1, 2)
                inner_env = eval_env._env._env._env._env
                all_img = []
                for xy, g_xy in zip(maze_states, goal_states):
                    inner_env.s_xy = xy
                    inner_env.g_xy = g_xy
                    img = (eval_env.render().astype(np.float32) / 255.0) - 0.5
                    all_img.append(img)
                # Revert environment
                eval_env.clear_plots()
                imgs = np.stack(all_img, 0)
                imgs = imgs.reshape([*elite_obs.shape[:2], 100,100,3]) # T x B x H x W x 3
                T,B,H,W,C = imgs.shape
                # want T,H,B,W,C
                imgs = imgs.transpose(0,2,1,3,4).reshape((T,H,B*W,C)) + 0.5
                metric = {f"top_{num_vis}_cem": imgs}
                logger.add(metric)
                logger.write()
                
        return vis_fn


    
    def make_sample_env_goals_fn(self, config, env):
        sample_env_goals_fn = None
        if config.task == 'hardumazefulldownscale' or 'demofetchpnp' in config.task:

            def sample_env_goals_fn(num_samples):
                all_goals = tf.convert_to_tensor(env.get_goals(), dtype=tf.float32)
                N = len(all_goals)
                goal_ids = tf.random.categorical(
                    tf.math.log([[1/N] * N]), num_samples)
                # tf.print("goal ids", goal_ids)
                return tf.gather(all_goals, goal_ids)[0]

        return sample_env_goals_fn


    def draw_demo_goal_distribution(self, logdir, env, train_dataset, logger):

        result_img_path = logdir / 'demo_goal_distribution.png'
        result_img_path_seed = logdir / 'demo_goal_distribution_seed.png' 

        ep_demogoal_rate_list = []
        ep_demogoal_rate_list_seed = []

        sample_rate = 1
        print(f"Sampling rate: {sample_rate}")

        None_count = 0

        selected_seed = None
        # Iterate through the training dataset
        for key, value in train_dataset.items():
            goal = value['goal'].tolist()[-1]
            demogoal_rate, seed = env.get_demogoal_index_rate(goal)

            if selected_seed is None:

                selected_seed = seed
                print(f"Selected seed: {selected_seed}")

            if demogoal_rate is not None:
                if random.random() < sample_rate:
                    ep_demogoal_rate_list.append(demogoal_rate)

                    if seed == selected_seed:
                        ep_demogoal_rate_list_seed.append(demogoal_rate)
                else:
                    continue
            else:
                None_count += 1
        
        # print(f"None demogoal rate count: {None_count}")

        # Prepare data for plotting
        episode_numbers = range(1, len(ep_demogoal_rate_list) + 1)
        episode_numbers_seed = range(1, len(ep_demogoal_rate_list_seed) + 1)  

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(episode_numbers, ep_demogoal_rate_list, color='red', s=1)  # s controls the size of the points
        plt.title('Demo Goal Distribution')
        plt.xlabel('Episode Number')
        plt.ylabel('Demo Goal Rate')
        plt.grid(True)
        plt.savefig(result_img_path)
        plt.close()

        # Plotting for ep_demogoal_rate_list_seed
        plt.figure(figsize=(8, 6))
        plt.scatter(episode_numbers_seed, ep_demogoal_rate_list_seed, color='red', s=1)
        plt.title('Demo Goal Distribution (One Seed)')
        plt.xlabel('Episode Number')
        plt.ylabel('Demo Goal Rate')
        plt.grid(True)
        plt.savefig(result_img_path_seed)
        plt.close()


        image_array = imageio.imread(result_img_path)[..., :3]
        image_array = np.expand_dims(image_array, axis=0) 
        image_array_seed = imageio.imread(result_img_path_seed)[..., :3]
        image_array_seed = np.expand_dims(image_array_seed, axis=0) 

        logger.image('goal_sample_rate_image', image_array)
        logger.image('goal_sample_rate_image_seed', image_array_seed)

        logger.write()

        return


    
    def train(self, env, eval_env, eval_fn, ep_render_fn, images_render_fn, space_explored_plot_fn, cem_vis_fn, obs2goal_fn, sample_env_goals_fn, config):

        
        logdir = pathlib.Path(config.logdir).expanduser()
        logdir.mkdir(parents=True, exist_ok=True)
        config.save(logdir / 'config.yaml')
        print(config, '\n')
        print('Logdir: ', logdir)

        if 'pointmaze' in config.task:
            video_fps = 10
        else:
            video_fps = 20

        
        outputs =  [
            common.TerminalOutput(),
            common.JSONLOutput(config.logdir),
            common.TensorBoardOutput(config.logdir, video_fps),
        ]

        
        
        replay = common.Replay(logdir / 'train_episodes', **config.replay)  # initialize replay buffer

        
        eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.dataset.length,
            maxlen=config.dataset.length))

        
        # initialize step counter
        step = common.Counter(replay.stats['total_steps'])

        
        # initialize episode counter
        num_eps = common.Counter(replay.stats['total_episodes'])

        
        num_algo_updates = common.Counter(0)

        
        logger = common.Logger(step, outputs, multiplier=config.action_repeat)  # initialize logger

        
        metrics = collections.defaultdict(list)  # minitialize metrics list

        

        
        should_train = common.Every(config.train_every)  # train every 5 steps


        
        should_report = common.Every(config.report_every)  # report every 1e4 steps

        
        should_video_train = common.Every(config.eval_every)
        # should_video_train = common.Every(2)

        
        should_eval = common.Every(config.eval_every)  # eval every 133 rollouts.

        
        should_ckpt = common.Every(config.ckpt_every)  # ckpt every X episodes.

        
        # how often to refresh goal picker distribution
        should_goal_update = common.Every(config.goal_update_every)

        
        
        should_gcp_rollout = common.Every(config.gcp_rollout_every)

        
        should_exp_rollout = common.Every(config.exp_rollout_every)

        
        should_two_policy_rollout = common.Every(config.two_policy_rollout_every)

        should_jsrl_rollout = common.Every(config.jsrl_rollout_every)

        should_cem_plot = common.Every(config.eval_every) # show image every time it evaluates

        
        
        if config.if_egc_env_sample:
            should_env_gcp_rollout = common.Every(config.env_gcp_rollout_every)

        self.next_ep_video = False

        
        
        

        self.time_point = time.time()
        def per_episode(ep, mode):

            current_time_point = time.time()
            episode_time = current_time_point - self.time_point
            self.time_point = current_time_point
            logger.scalar('ep_time_spend(s)', round(episode_time, 2))
            if mode == 'train':
                remaining_time = (config.steps - step.value)//config.time_limit * episode_time
                remaining_hours = remaining_time // 3600
                logger.scalar('remaining_time(h)', remaining_hours)

            
            length = len(ep['reward']) - 1
            score = float(ep['reward'].astype(np.float64).sum())
            print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
            logger.scalar(f'{mode}_return', score)
            logger.scalar(f'{mode}_length', length)

            if config.goal_strategy == 'Demo_goal_Planner' and mode == 'train':

                try:
                    logger.scalar(f'demo_wm_error', my_GC_goal_picker.goal_strategy.demo_wm_error_metric)
                except:
                    pass

            for key, value in ep.items():
                if re.match(config.log_keys_sum, key):
                    logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
                if re.match(config.log_keys_mean, key):
                    logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
                if re.match(config.log_keys_max, key):
                    logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())

            if (should_video_train(num_eps) or self.next_ep_video) and mode == 'train':

                # if self.next_ep_video:
                #     self.next_ep_video = False
                # else:
                #     self.next_ep_video = True

                if ep_render_fn is None and 'none' not in config.log_keys_video:
                    for key in config.log_keys_video:
                        logger.video(f'{mode}_policy_{key}', ep[key])

                elif ep_render_fn is not None:
                                    
                    video = ep_render_fn(env, ep)
                    if video is not None:
                        label = ep['label'][0]
                        logger.video(f'{mode}_policy_{config.state_key}_{label}', video)
            
            
                        
            # if mode == 'train':

            # #     print(ep['goal'][0])
            # #     self.cluster_ep_idx += 1

            #     train_ep_video = ep_render_fn(env, ep)

            
            #     train_ep_video = np.squeeze(train_ep_video) 

            
            # #     print(train_ep_video.shape)
            #     imageio.mimsave('ep_train.gif', train_ep_video, duration=0.2)

            # ==============================================================================

            _replay = dict(train=replay, eval=eval_replay)[mode]
            logger.add(_replay.stats, prefix=mode)
            logger.write()

        
        driver = common.GCDriver([env], config.goal_key, config)  
        
        driver.on_episode(lambda ep: per_episode(ep, mode='train'))
        driver.on_episode(lambda ep: num_eps.increment())

        driver.on_step(lambda tran, worker: step.increment())  
        driver.on_step(replay.add_step)
        driver.on_reset(replay.add_step)

        
        eval_driver = common.GCDriver([eval_env], config.goal_key, config)
        eval_driver.if_eval_driver = True
        eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
        eval_driver.on_episode(eval_replay.add_episode)

        if config.if_assign_centroids:
            centroids_sample_driver = common.GCDriver([eval_env], config.goal_key, config)  

        print('Create agent.')

        
        agnt = gc_agent.GCAgent(config, env.obs_space, env.act_space, step, obs2goal_fn, sample_env_goals_fn)

        env._env.agnt = agnt

        
        train_agent = common.CarryOverState(agnt.train)

        
        train_gcp = common.CarryOverState(agnt.train_gcp)
        if config.if_actor_gs:
            train_gcpolicy = partial(agnt.policy_gs, mode='train')
            eval_gcpolicy = partial(agnt.policy_gs, mode='eval')
        else:
            train_gcpolicy = partial(agnt.policy, mode='train')
            eval_gcpolicy = partial(agnt.policy, mode='eval')


        # space_explored_plot======================================================================================
        should_space_explored_plot = common.Every(config.eval_every) # show sapce explored image every time it evaluates
        def space_explored_plot(): # define the plot function after agent has been defined.
            if should_space_explored_plot(num_eps) and space_explored_plot_fn != None:
                from time import time
                plt.cla()
                plt.clf()
                start = time()
                space_explored_plot_fn(eval_env, agnt=agnt, complete_episodes=replay._complete_eps, logger=logger, ep_subsample=1, step_subsample=1)
                print("plotting took ", time() - start)
                logger.write()
        driver.on_episode(lambda ep: space_explored_plot())
        # space_explored_plot======================================================================================


        if_BC_pretrain = config.if_BC_pretrain
        # Modem Pretrain Policy and World Model(phase 1) and sample trajectories use pretrained policy(phase 2)

        if if_BC_pretrain:

            demo_replay = common.Replay(logdir / 'demo_episodes', **config.demo_replay)  # initialize replay buffer
            demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))
            demo_batch_data = next(demo_dataset)

            if (logdir / 'variables.pkl').exists():

                
                dataset = iter(replay.dataset(**config.dataset))  

                if config.replay.sample_recent:
                    recent_dataset = iter(replay.recent_dataset(**config.dataset))  

                
                # for train vid pred.
                report_dataset = iter(replay.dataset(**config.dataset))

                train_agent(next(dataset))

                print('Found existing checkpoint.')
                agnt.agent_load(logdir)

            else:
                print("BC Pretrain")
                pretrain_epoch = 10000
                print("BC Pretrain epoch: ", pretrain_epoch)
                self.k = 0.5

                # Pretrain World Model(phase 1)
                for _ in tqdm(range(pretrain_epoch), desc='Demo Pretrain'):

                    _demodata = next(demo_dataset)
                    # agnt.train_wm(_demodata)
                    train_agent(_demodata)

                for _ in tqdm(range(pretrain_epoch), desc='Agent demo BC Pretrain'):

                    agnt._task_behavior.BC_train(agnt.wm)


                # sample trajectories use pretrained policy(phase 2)
                phase2_tra_num = config.prefill
                # phase2_tra_num = 5

                for _ in range(phase2_tra_num):
                    
                    driver(train_gcpolicy, episodes=1)


                
                dataset = iter(replay.dataset(**config.dataset))  

                if config.replay.sample_recent:
                    recent_dataset = iter(replay.recent_dataset(**config.dataset))  

                
                # for train vid pred.
                report_dataset = iter(replay.dataset(**config.dataset))

                train_agent(next(dataset))


                # for _ in range(pretrain_epoch):

                #     _demodata = next(demo_dataset)
                #     _sampledata = next(dataset)

                #     assert _demodata['observation'].shape[0] == _sampledata['observation'].shape[0]

                
                #     demo_split = int(self.k * _demodata['observation'].shape[0])
                #     sample_split = int(_sampledata['observation'].shape[0] - demo_split)

                
                #     demo_part = {key: value[:demo_split] for key, value in _demodata.items()}
                #     sample_part = {key: value[:sample_split] for key, value in _sampledata.items()}

                #     keys_to_process = ['observation', 'action', 'reward', 'goal', 'is_first', 'is_last', 'is_terminal']


                #     _traindata = {}
                #     for key in keys_to_process:

                #         if key == 'is_terminal':

                #             demo_part[key] = tf.cast(demo_part[key], tf.float32)

                #         _traindata[key] = tf.concat([demo_part[key], sample_part[key]], axis=0)


                
                #     train_agent(_traindata)

        else:

            
            prefill = max(0, config.prefill - replay.stats['total_episodes'])

            random_agent = common.RandomAgent(env.act_space)
            if prefill:
                print(f'Prefill dataset ({prefill} episodes).')
                driver(random_agent, episodes=prefill)
                driver.reset()

            
            dataset = iter(replay.dataset(**config.dataset))  

            if config.replay.sample_recent:
                recent_dataset = iter(replay.recent_dataset(**config.dataset))  

            
            # for train vid pred.
            report_dataset = iter(replay.dataset(**config.dataset))

            print("Test Train")
            test_batch_data = next(dataset)
            train_agent(test_batch_data)
            print("Test Train Done")

            if (logdir / 'variables.pkl').exists():
                print('Found existing checkpoint.')
                agnt.agent_load(logdir)

            else:
                print('Pretrain agent.')
                for _ in range(config.pretrain):
                    for i in range(config.gcp_train_factor - 1):
                        train_gcp(next(gcp_dataset))
                    train_agent(next(dataset))


        # egc dataset
        if config.if_egc_env_sample:
            if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:
                self.cluster_assign_egc_goal_index = 1
                egc_dataset_1 = iter(replay.recent_dataset_specific_label(**config.dataset, label='egc1'))
                egc_dataset_2 = iter(replay.recent_dataset_specific_label(**config.dataset, label='egc2'))
                egc_dataset_3 = iter(replay.recent_dataset_specific_label(**config.dataset, label='egc3'))

            else:
                egc_dataset = iter(replay.recent_dataset_specific_label(**config.dataset, label='egc'))  

        if config.gcp_train_factor > 1:
            gcp_dataset = iter(replay.dataset(**config.dataset))  

        if config.replay.sample_recent:
            recent_dataset = iter(replay.recent_dataset(**config.dataset))  

            if config.gcp_train_factor > 1:
                
                recent_gcp_dataset = iter(replay.recent_dataset(**config.dataset))

        if config.if_goal_optimizer:

            agnt.goal_optimizer.train(next(dataset))


        
        def train_step(tran, worker):

            if should_train(step):
                # start_time = time()
                # data_duration = 0
                # train_duration = 0

                
                for _ in range(config.train_steps):

                    _data = next(dataset)
                    mets = train_agent(_data)
                    [metrics[key].append(value) for key, value in mets.items()]

                    # if config.if_goal_optimizer and self.if_train_goal_optimizer:

                    #     agnt.goal_optimizer.train(_data)

                    # if config.if_use_demo:

                    #     _demodata = next(demo_dataset)
                    #     train_agent(_demodata)

                    
                    # if config.train_cluster_use_Normal:
                    #     mets = train_agent(_data, train_cluster=True)
                    # else:
                    #     mets = train_agent(_data)

                    # [metrics[key].append(value) for key, value in mets.items()]
                    
                    
                    # for i in range(config.gcp_train_factor - 1):
                    #     mets = train_gcp(next(gcp_dataset))
                    #     [metrics[key].append(value) for key, value in mets.items()]

                    
                    if config.replay.sample_recent:
                        _sampledata = next(recent_dataset)

                        if config.demo_over_sample:

                            _demodata = next(demo_dataset)

                            assert _demodata['observation'].shape[0] == _sampledata['observation'].shape[0]

                            
                            self.k = max(0.75 - 0.5 * step.value / config.steps, 0.25)
                            demo_split = int(self.k * _demodata['observation'].shape[0])
                            sample_split = int(_sampledata['observation'].shape[0] - demo_split)

                            
                            demo_part = {key: value[:demo_split] for key, value in _demodata.items()}
                            sample_part = {key: value[:sample_split] for key, value in _sampledata.items()}

                            keys_to_process = ['observation', 'action', 'reward', 'goal', 'is_first', 'is_last', 'is_terminal']


                            _traindata = {}
                            for key in keys_to_process:

                                if key == 'is_terminal':

                                    demo_part[key] = tf.cast(demo_part[key], tf.float32)

                                _traindata[key] = tf.concat([demo_part[key], sample_part[key]], axis=0)


                            
                            mets = train_agent(_traindata)
                        
                        else:

                            # time_record = time.time()

                            mets = train_agent(_sampledata)

                            # print("train_agent time", time.time() - time_record)

                        [metrics[key].append(value) for key, value in mets.items()]

                        
                        for i in range(config.gcp_train_factor - 1):
                            mets = train_gcp(next(recent_gcp_dataset))
                            [metrics[key].append(value)
                            for key, value in mets.items()]


                    if config.if_egc_env_sample and config.train_cluster_use_egc:

                        if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                            if self.cluster_assign_egc_goal_index == 1:

                                _egc_data = next(egc_dataset_1)

                            elif self.cluster_assign_egc_goal_index == 2:

                                _egc_data = next(egc_dataset_2)
                            
                            elif self.cluster_assign_egc_goal_index == 3:

                                _egc_data = next(egc_dataset_3)

                            else:
                                raise ValueError("Wrong value of self.cluster_assign_egc_goal_index!")


                        else:
                            _egc_data = next(egc_dataset)
                            
                            

                        
                        mets = train_agent(_egc_data, train_cluster=True)


            
            if should_report(step):
                for name, values in metrics.items():
                    logger.scalar(name, np.array(values, np.float64).mean())
                    metrics[name].clear()

                
                logger.add(agnt.report(next(report_dataset), eval_env))
                logger.write(fps=True)


        
        driver.on_step(train_step)

        
        def eval_agent():
            if should_eval(num_eps):
                print('Start evaluation.')
                print("demo_obs_count_dict: ", env.demo_obs_count_dict)
                print("Drawing demo goal sampled distribution")
                self.draw_demo_goal_distribution(logdir, env, train_dataset=replay._complete_eps, logger=logger)

                sys.stdout.flush()
                if config.if_goal_optimizer and self.if_train_goal_optimizer:
                    eval_fn(eval_driver, eval_gcpolicy, logger, goal_optimizer=agnt.goal_optimizer)
                else:
                    eval_fn(eval_driver, eval_gcpolicy, logger)
                agnt.agent_save(logdir)

                # show cluster centroids
                if images_render_fn is not None and config.goal_strategy == "Cluster_goal_Planner":

                    def get_decoded_centroids(agnt, config):

                        centroids = agnt.wm.cluster.centroids()

                        # print(centroids)
                        # print(centroids.shape)

                        centroids = tf.convert_to_tensor(centroids.numpy(), dtype=agnt.wm.dtype)

                        # centroid_1_embed = centroids[0]

                        state = None
                        if state is None:
                            latent = agnt.wm.rssm.initial(config.cluster['n_latent_landmarks'])
                            action = tf.zeros((config.cluster['n_latent_landmarks'],) + agnt.act_space.shape)
                            state = latent, action

                        latent, action = state

                        # print("latent", latent)
                        # print(centroids)

                        latent, _ = agnt.wm.rssm.obs_step(latent, action, centroids, True, True)

                        decoder = agnt.wm.heads['decoder']

                        feat = agnt.wm.rssm.get_feat(latent)

                        # print("feat", feat)

                        centroids_decoded_dist = decoder(feat)

                        centroids_decoded = centroids_decoded_dist[agnt.state_key].mean()

                        centroids_decoded = centroids_decoded.numpy()

                        return centroids_decoded
                    
                    decoded_centroids_list = get_decoded_centroids(agnt, config)
                    all_images = images_render_fn(eval_env, decoded_centroids_list)
                    logger.image('Cluster Centroids', all_images)
                
            if should_ckpt(num_eps):
                print('Checkpointing.')
                agnt.agent_save(logdir)

        
        def vis_fn(elite_inds, elite_samples, seq, wm):
            if should_cem_plot(num_eps) and cem_vis_fn is not None:
                cem_vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger)


        print("Goal Pick Strategy for Exploration: ", config.goal_strategy)
        my_GC_goal_picker = gc_goal_picker.GC_goal_picker(config, agnt, replay, dataset, env, obs2goal_fn, sample_env_goals_fn, vis_fn)

        get_goal_fn = my_GC_goal_picker.get_goal_fn


        def assign_cluster_centrods():

            print("==========================================================clusters update==========================================================")

            if config.if_assign_use_batch:

                    if config.centrods_assign_strategy == 'egc':

                        if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                            self.cluster_assign_egc_goal_index = random.randint(1, 3)

                            if self.cluster_assign_egc_goal_index == 1:

                                _egc_data = next(egc_dataset_1)

                            elif self.cluster_assign_egc_goal_index == 2:

                                _egc_data = next(egc_dataset_2)
                            
                            elif self.cluster_assign_egc_goal_index == 3:

                                _egc_data = next(egc_dataset_3)

                            else:
                                raise ValueError("Wrong value of self.cluster_assign_egc_goal_index!")
                            
                        else:
                            _egc_data = next(egc_dataset)  # should all have label egc

                        _egc_data[agnt.state_key] = _egc_data[agnt.state_key].numpy()
                        _egc_data[agnt.state_key] = _egc_data[agnt.state_key].reshape(-1, _egc_data[agnt.state_key].shape[-1])
                        ep = _egc_data
                        

                    elif config.centrods_assign_strategy == 'exp':

                        _data = next(dataset)  # should all have label Normal

                        _data[agnt.state_key] = _data[agnt.state_key].numpy()
                        _data[agnt.state_key] = _data[agnt.state_key].reshape(-1, _data[agnt.state_key].shape[-1])
                        ep = _data

            else:

                centroids_sample_driver.reset()

                if config.centrods_assign_strategy == 'egc':

                    if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                        
                        centroids_sample_driver(eval_gcpolicy, if_multi_3_blcok_training_goal = config.if_multi_3_blcok_training_goal, episodes=1, label='egc')

                        self.cluster_assign_egc_goal_index = centroids_sample_driver.training_goal_index
                    
                    else:
                        centroids_sample_driver(eval_gcpolicy, episodes=1, label='egc')

                elif config.centrods_assign_strategy == 'exp':

                    centroids_sample_driver(train_gcpolicy, expl_policy, get_goal_fn, episodes=1, goal_time_limit=goal_time_limit, goal_checker=temporal_dist)
                
                ep = centroids_sample_driver._eps[0]
                # print(len(ep))
                ep = {k: centroids_sample_driver._convert([t[k] for t in ep]) for k in ep[0]}

            agnt.wm.assign_cluster_centroids(data = ep, space = config.centroids_assign_space)

        def update_goal_strategy(*args):
            if should_goal_update(num_eps):
                if config.goal_strategy == "Greedy":
                    my_GC_goal_picker.goal_strategy.update_buffer_priorities()

                elif config.goal_strategy in {"MEGA", "Skewfit"}:
                    my_GC_goal_picker.goal_strategy.update_kde()
                
                elif config.goal_strategy == "SubgoalPlanner":
                    # goal strategy will search for new distribution next time we sample.
                    my_GC_goal_picker.goal_strategy.will_update_next_call = True

                elif config.goal_strategy == "Cluster_goal_Planner" and config.if_assign_centroids:
                    assign_cluster_centrods()
                
                elif config.goal_strategy == "Demo_goal_Planner":

                    my_GC_goal_picker.goal_strategy.learning_rate = step.value / config.steps
                    my_GC_goal_picker.goal_strategy.if_eval_peg = True
                    # for key, value in my_GC_goal_picker.goal_strategy.search_repo.items():
                    #     my_GC_goal_picker.goal_strategy.search_repo[key][0] = True
                       

        driver.on_episode(lambda ep: update_goal_strategy())  
        driver.on_episode(lambda ep: env._env.count_demo_obs())  

        goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)

        
        def temporal_dist(obs):
            obs = obs.copy()
            obs = {key: value for key, value in obs.items() if key not in ['env_states']}
            obs = tf.nest.map_structure(
                lambda x: tf.expand_dims(tf.tensor(x), 0), obs)
            dist = agnt.temporal_dist(obs).numpy().item()
            # success = dist < config.subgoal_threshold
            success = False
            metric = {"subgoal_dist": dist, "subgoal_success": float(success)}
            return success, metric

        
        def expl_policy(obs, state, **kwargs):

            
            actions, state = agnt.expl_policy(obs, state, mode='train')

            
            if config.go_expl_rand_ac:
                actions, _ = random_agent(obs)

            return actions, state

        # ======================================================================================================

        while step < config.steps:

            if config.if_goal_optimizer and step > config.goal_optimizer_start_step:
                self.if_train_goal_optimizer = True
                goal_optimizer = agnt.goal_optimizer
            else:
                goal_optimizer = None


            
            logger.write()

            # alternate between these 3 types of rollouts.
            """ 1. train: run goal cond. policy for entire rollout"""
            if should_gcp_rollout(num_algo_updates):
                driver(train_gcpolicy, get_goal=get_goal_fn, episodes=1)
                eval_agent()

            """ 2. expl: run expl policy for entire rollout """
            if should_exp_rollout(num_algo_updates):
                driver(expl_policy, episodes=1)
                eval_agent()

            """ 3. 2pol: run goal cond. and then expl policy."""
            if should_two_policy_rollout(num_algo_updates):

                
                driver(train_gcpolicy, expl_policy, get_goal_fn, goal_optimizer=goal_optimizer, episodes=1,
                    goal_time_limit=goal_time_limit, goal_checker=temporal_dist)
                eval_agent()
                # sys.exit()

            if should_jsrl_rollout(num_algo_updates):

                goal_time_limit = round((0.8 - step.value / config.steps) * config.time_limit)
                driver(expl_policy, train_gcpolicy, get_goal_fn, goal_optimizer=goal_optimizer, episodes=1,
                    goal_time_limit=goal_time_limit, goal_checker=temporal_dist)
                eval_agent()
                # sys.exit()

            if config.if_egc_env_sample and should_env_gcp_rollout(num_algo_updates):

                if 'demofetchpnp' in config.task and config.if_multi_3_blcok_training_goal:

                    driver(train_gcpolicy, goal_optimizer=goal_optimizer, episodes=3, if_multi_3_blcok_training_goal = config.if_multi_3_blcok_training_goal, label='egc')

                else:

                    driver(train_gcpolicy, goal_optimizer=goal_optimizer, episodes=1, label='egc')

                eval_agent()

            num_algo_updates.increment()
        # ======================================================================================================


    def main(self):
        """
        Pass in the config setting(s) you want from the configs.yaml. If there are multiple
        configs, we will override previous configs with later ones, like if you want to add
        debug mode to your environment.

        To override specific config keys, pass them in with --key value.

        python examples/run_goal_cond.py --configs <setting 1> <setting 2> ... --foo bar

        Examples:
            Normal scenario
                python examples/run_goal_cond.py --configs mega_fetchpnp_proprio
            Debug scenario
                python examples/run_goal_cond.py --configs mega_fetchpnp_proprio debug
            Override scenario
                python examples/run_goal_cond.py --configs mega_fetchpnp_proprio --seed 123
        """
        
        config = self.Set_Config()

        self.config = config

        
        env = self.make_env(config)

        print("====================================")
        print("Environment successfully creat")
        print("Observation space:", env.obs_space)
        print("Action Space:", env.act_space)
        print("====================================")
        eval_env = self.make_env(config, if_eval=True)

        # obs = env.reset()

        
        sample_env_goals_fn = self.make_sample_env_goals_fn(config, eval_env)
        eval_fn = self.make_eval_fn(config)
        ep_render_fn = self.make_ep_render_fn(config)


        space_explored_plot_fn = self.make_space_explored_plot_fn(config)
        cem_vis_fn = self.make_cem_vis_fn(config)
        obs2goal_fn = self.make_obs2goal_fn(config)
        images_render_fn = self.make_images_render_fn(config)

        
        if_run_eagerly = self.config.if_run_eagerly
        if if_run_eagerly:
            warnings.warn("Run code in eager mode. This will be slow !!!")
            
        tf.config.run_functions_eagerly(if_run_eagerly)  
        # tf.data.experimental.enable_debug_mode(not config.jit)
        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message

        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)

        if gpus:
            
            
            gpu_index = self.config.gpu_index
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            print(f"Using GPU: {gpus[gpu_index].name}")
        else:
            print("No GPU devices found")

        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        if config.precision == 16:
            from tensorflow.keras.mixed_precision import experimental as prec
            prec.set_policy(prec.Policy('mixed_float16'))

        
        self.train(env, eval_env, eval_fn, ep_render_fn, images_render_fn, space_explored_plot_fn, cem_vis_fn, obs2goal_fn, sample_env_goals_fn, config)


if __name__ == "__main__":

    My_Method = Method()
    My_Method.main()









