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
import nor_agent
import common
import dreamerv2_demo.gc_goal_picker as gc_goal_picker


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


class Method: 

    def __init__(self):

        pass

    
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
            new_keys = ['if_goal_optimizer', 'goal_optimizer_start_step', 'classifier', 'if_run_eagerly']
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

        if config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1" or config.task == "PickAndPlace" or config.task == "FetchPush-v2" or config.task == "FetchSlide-v2" or "HandManipulateBlock" in config.task or "HandManipulatePen" in config.task or "HandManipulateEgg" in config.task or "Adroit" in config.task or "meta" in config.task:
            
            
            import gymnasium as gym

            try:
                import mani_skill2.envs
            except:
                pass
            # from mani_skill2.utils.sapien_utils import vectorize_pose

            if config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0":

                # supported modes: ['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 'pd_ee_delta_pose_align', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
                env = gym.make(config.task, obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", max_episode_steps=config.time_limit)

            elif config.task == 'OpenCabinetDrawer-v1':

                env = gym.make("OpenCabinetDrawer-v1", reward_mode = "sparse", obs_mode = "state", model_ids = ["1000"], render_mode="rgb_array", max_episode_steps=config.time_limit)
                
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

                return e
            
            env = gymnasium_env(env)



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

        elif config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1":

            def episode_render_fn(env, ep):

                
                all_img = []

                # print(ep['goal'][0])

                goal_render_state = env.get_demogoal_render_state(env.goal_idx, ep['goal'][0])

                if goal_render_state is None:
                    goal_img = None
                else:
                    env.unwrapped.set_state(goal_render_state)
                    goal_img = env.render()
                
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
                

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]


                return ep_img

        elif "meta" in config.task:

            def episode_render_fn(env, ep):

                
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

            def episode_render_fn(env, ep):

                env.reset()
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

                goal_idx = ep['goal_idx'][0]
                env._env.set_goal_idx(goal_idx)

                env.reset()
                
                all_img = []

                new_ep = []

                for action in ep['action']:

                    action = {'action': action}
                    
                    obs = env.step(action)

                    new_ep.append(obs)

                    all_img.append(env.render())

                ep_img = np.stack(all_img, 0)

                T = ep_img.shape[0]

                
                if config.time_limit+1 - T > 0:
                    ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
                    ep_img[-((config.time_limit+1) - T):] = all_img[-1]

                return ep_img
            
            def episode_render_fn(env, ep):

                env.reset()
                
                all_img = []

                for obs, goal in zip(ep['observation'], ep['goal']):

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


        elif config.task == "PegInsertionSide-v0" or config.task == "StackCube-v0" or "Adroit" in config.task or config.task == 'OpenCabinetDrawer-v1' or config.task == "PushChair-v1" or "meta" in config.task:


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
                
                random_goal_indices = np.random.choice(eval_goal_num, size=50, replace=False)

            

                
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
                            ep_video = episode_render_fn(env, ep)
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
                            ep_video = episode_render_fn(env, ep)
                            all_ep_video.append(ep_video[None])  # 1 x T x H x W x C


                all_ep_video = np.concatenate(all_ep_video, 3)
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
                                ep_video = episode_render_fn(env, ep)
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
                        ep_video = episode_render_fn(env, ep)
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

                eval_fn_with_demo_seed_reset(driver, eval_policy, logger)
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


    
    def train(self, env, eval_env, eval_fn, ep_render_fn, config):

        
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


        if config.if_use_demo:

            demo_replay = common.Replay(logdir / 'demo_episodes', **config.demo_replay)  # initialize replay buffer
            demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))
            demo_batch_data = next(demo_dataset)

        
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

        
        should_eval = common.Every(config.eval_every)  # eval every 133 rollouts.

        
        should_ckpt = common.Every(config.ckpt_every)  # ckpt every X episodes.

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

            for key, value in ep.items():
                if re.match(config.log_keys_sum, key):
                    logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
                if re.match(config.log_keys_mean, key):
                    logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
                if re.match(config.log_keys_max, key):
                    logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())

            if (should_video_train(num_eps) or self.next_ep_video) and mode == 'train':

                if self.next_ep_video:
                    self.next_ep_video = False
                else:
                    self.next_ep_video = True

                if ep_render_fn is None and 'none' not in config.log_keys_video:
                    for key in config.log_keys_video:
                        logger.video(f'{mode}_policy_{key}', ep[key])

                elif ep_render_fn is not None:
                                    
                    video = ep_render_fn(env, ep)
                    if video is not None:
                        if 'label' in ep.keys():
                            label = ep['label'][0]
                            logger.video(f'{mode}_policy_{config.state_key}_{label}', video)
                        else:
                            logger.video(f'{mode}_policy_{config.state_key}', video)
            

            _replay = dict(train=replay, eval=eval_replay)[mode]
            logger.add(_replay.stats, prefix=mode)
            logger.write()

        
        driver = common.Driver([env])  
        
        driver.on_episode(lambda ep: per_episode(ep, mode='train'))
        driver.on_episode(lambda ep: num_eps.increment())

        driver.on_step(lambda tran, worker: step.increment())  
        driver.on_step(replay.add_step)
        driver.on_reset(replay.add_step)

        
        eval_driver = common.Driver([eval_env])
        eval_driver.if_eval_driver = True
        eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
        eval_driver.on_episode(eval_replay.add_episode)

        print('Create agent.')

        
        agnt = nor_agent.Agent(config, env.obs_space, env.act_space, step)
        train_policy = partial(agnt.policy, mode='train')
        eval_policy = partial(agnt.policy, mode='eval')
        
        train_agent = common.CarryOverState(agnt.train)
 

        if_Modem = False
        # Modem Pretrain Policy and World Model(phase 1) and sample trajectories use pretrained policy(phase 2)
        if (logdir / 'variables.pkl').exists():
            print('Found existing checkpoint.')
            agnt.agent_load(logdir)

        else:

            if if_Modem:
                print("Modem Phase 1")
                Modem_pretrain_epoch = 10000
                print("BC Pretrain epoch: ", Modem_pretrain_epoch)
                self.k = 0.5

                # Pretrain Policy and World Model(phase 1)
                for _ in range(Modem_pretrain_epoch):

                    _demodata = next(demo_dataset)
                    train_agent(_demodata)

                for _ in range(Modem_pretrain_epoch):

                    agnt._task_behavior.BC_train(agnt.wm)


                # sample trajectories use pretrained policy(phase 2)
                print("Modem Phase 2")
                phase2_tra_num = 100

                for _ in range(phase2_tra_num):
                    
                    driver(train_policy, episodes=1)

                
                dataset = iter(replay.dataset(**config.dataset))  

                if config.replay.sample_recent:
                    recent_dataset = iter(replay.recent_dataset(**config.dataset))  

                
                # for train vid pred.
                report_dataset = iter(replay.dataset(**config.dataset))

                for _ in range(Modem_pretrain_epoch):

                    _demodata = next(demo_dataset)
                    _sampledata = next(dataset)

                    assert _demodata['observation'].shape[0] == _sampledata['observation'].shape[0]

                    
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


                    
                    train_agent(_traindata)
                
                print("Modem Phase 3")

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

                train_agent(next(dataset))
                
                print('Pretrain agent.')
                for _ in range(config.pretrain):

                    train_agent(next(dataset))


        
        def train_step(tran, worker):

            if should_train(step):
                # start_time = time()
                # data_duration = 0
                # train_duration = 0

                
                for _ in range(config.train_steps):


                    if if_Modem:

                        _demodata = next(demo_dataset)
                        _sampledata = next(dataset)

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
                        _data = next(dataset)

                        
                        mets = train_agent(_data)

                    [metrics[key].append(value) for key, value in mets.items()]

                    
                    # if config.replay.sample_recent:
                    #     _data = next(recent_dataset)
                    #     mets = train_agent(_data)
                    #     [metrics[key].append(value) for key, value in mets.items()]

            
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
                if config.if_goal_optimizer and self.if_train_goal_optimizer:
                    eval_fn(eval_driver, eval_policy, logger, goal_optimizer=agnt.goal_optimizer)
                else:
                    eval_fn(eval_driver, eval_policy, logger)
                agnt.agent_save(logdir)
          
            if should_ckpt(num_eps):
                print('Checkpointing.')
                agnt.agent_save(logdir)


        
        def expl_policy(obs, state, **kwargs):

            
            actions, state = agnt.expl_policy(obs, state, mode='train')

            
            # if config.go_expl_rand_ac:
            #     actions, _ = random_agent(obs)

            return actions, state

        # ======================================================================================================

        while step < config.steps:

            
            logger.write()

            eval_agent()
            driver(train_policy, episodes=1)
            eval_agent()

            num_algo_updates.increment()
        # ======================================================================================================


    def main(self):
        
        config = self.Set_Config()

        self.config = config

        
        env = self.make_env(config)
        eval_env = self.make_env(config, if_eval=True)

        # obs = env.reset()

        
        eval_fn = self.make_eval_fn(config)
        ep_render_fn = self.make_ep_render_fn(config)


        
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

        
        self.train(env, eval_env, eval_fn, ep_render_fn, config)


if __name__ == "__main__":

    My_Method = Method()
    My_Method.main()









