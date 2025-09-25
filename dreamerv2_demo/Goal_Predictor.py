import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import gymnasium as gym
import imageio
from PIL import Image
import h5py
import json

import yaml
import pathlib
from dreamerv2_demo import common
import sys
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Set the specific GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(0)  # Use GPU 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_demo_trajectories(task, task_demo_path, if_eval=False, if_random_seeds=False, random_seeds_num=50, image_size=(100, 100)):


    if task == "PegInsertionSide-v0" or task == "StackCube-v0" or task == "PickCube-v0" or task == "PullCubeTool-v1"  or task == "OpenCabinetDrawer-v1" or task == "PushChair-v1":

        if task == "PegInsertionSide-v0":

            tra_json_path = task_demo_path.replace('.h5', '.json')

            if not if_random_seeds:
                if if_eval:
                    seed_list = [8, 9, 11, 12]
                else:
                    seed_list = [0, 1, 2, 3, 5, 7, 8, 9]
            else:
                seed_list = []

        if task == "StackCube-v0" or task == "PickCube-v0":

            tra_json_path = task_demo_path.replace('.h5', '.json')

            if not if_random_seeds:
                if if_eval:
                    seed_list = [8, 9, 11, 12]
                else:
                    seed_list = [0, 1, 2, 3, 5, 7, 8, 9]
            else:
                seed_list = []

        if task == "PullCubeTool-v1" :

            tra_json_path = task_demo_path.replace('.h5', '.json')

            if not if_random_seeds:
                if if_eval:
                    seed_list = []
                else:
                    seed_list = []
            else:
                seed_list = []


        elif task == "OpenCabinetDrawer-v1":

            tra_json_path = 'Demo/OpenDrawer/demos/v0/rigid_body/OpenCabinetDrawer-v1/1000/link_1/trajectory.json'

            if not if_random_seeds:
                if if_eval:
                    seed_list = [2340255427, 3638918503, 1819583497, 2678185683]
                else:
                    seed_list = [2357136044, 2546248239, 3071714933, 3626093760, 2588848963, 3684848379, 2340255427, 3638918503]
            else:
                
                seed_list = []             

        elif task == "PushChair-v1":
            tra_json_path = 'Demo/PushChair/demo_data/3001/trajectory.json'

            if not if_random_seeds:

                if if_eval:
                    seed_list = [2051556033, 305097549, 3576074995, 4110950085, 3342157822]
                else:
                    seed_list = [3071714933, 3684848379, 2340255427, 3638918503, 2678185683, 243580376, 2051556033, 305097549]
            else:
                seed_list = [] 

        with open(tra_json_path, 'r') as file:
            data = json.load(file)

            
            episode_id_to_seed = {}
            # episode_id_to_reset_kwargs = {}
            all_seeds_list = []

            
            for episode in data['episodes']:
                episode_id = episode['episode_id']
                episode_seed = episode['episode_seed']
                episode_id_to_seed[episode_id] = episode_seed
                # episode_id_to_reset_kwargs[episode_id] = episode['reset_kwargs']
                all_seeds_list.append(episode_seed)


        # import gymnasium as gym

        # env = gym.make("PushChair-v1", **data['env_info']['env_kwargs'], render_mode="rgb_array")
        # print("Observation space: ", env.observation_space)
        # print("Action space: ", env.action_space)

        demo_file_name= task_demo_path.split('/')[-1]
        # controller = demo_file_name.split('.')[2]

        if task == "PullCubeTool-v1":
            try:
                import gymnasium as gym
                import mani_skill.envs
                from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
            except:
                pass

            # supported controller modes: ['pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 'pd_ee_delta_pose_align', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel']
            env = gym.make(task, **data['env_info']['env_kwargs'])
            env = CPUGymWrapper(env)
        else:
            
            try:
                import gymnasium as gym
                import mani_skill2.envs
            except:
                pass
            
            if task == "OpenCabinetDrawer-v1":
                env = gym.make(task, **data['env_info']['env_kwargs'], render_mode="rgb_array")
            else:
                env = gym.make(task, **data['env_info']['env_kwargs'])
                
        all_demo_trajectories = []
        find_seed_list = []
        
        with h5py.File(task_demo_path, 'r') as f:

            # print(f.keys())
            for trajectory in f.keys():

                episode_index = int(trajectory.split('_')[-1])
                demo_seed = episode_id_to_seed[episode_index]

                if demo_seed not in seed_list and not if_random_seeds:
                    continue

                if len(find_seed_list) >= random_seeds_num:
                
                    break

                trajectory = f[trajectory]

                tra_dict = {}

                for key in trajectory.keys():
                    # print(key)
                    # print(trajectory[key].shape)

                    # print(trajectory)
                    if key == 'actions':

                        key_data = np.array(trajectory[key])

                        initial_action = np.zeros((1, key_data.shape[1]))

                        key_data = np.concatenate((initial_action, key_data), axis=0)

                    elif key == 'success':

                        if trajectory[key][-1] == 0:
                            break

                        key_data = np.array(trajectory[key])

                        initial_success = np.array([0])

                        key_data = np.concatenate((initial_success, key_data), axis=0)

                    else:
                        key_data = np.array(trajectory[key])

                    tra_dict[key] = key_data

                tra_dict['observation'] = tra_dict['obs']
                tra_dict['action'] = tra_dict['actions']

                tra_dict['goal'] = np.repeat(tra_dict['obs'][-1].reshape(1, tra_dict['obs'].shape[1]), tra_dict['obs'].shape[0], axis=0)

                tra_dict['reward'] = np.zeros(tra_dict['success'].shape, dtype=np.float32)

                tra_dict['is_first'] = np.zeros_like(tra_dict['success'], dtype=bool)
                tra_dict['is_first'][0] = True

                tra_dict['is_last'] = np.zeros_like(tra_dict['success'], dtype=bool)
                tra_dict['is_last'][-1] = True

                tra_dict['is_terminal'] = tra_dict['success']

                tra_dict['demo_seed'] = np.zeros_like(tra_dict['success'], dtype=int)
                tra_dict['demo_seed'][:] = demo_seed

                del tra_dict['obs']
                del tra_dict['actions']
                success = tra_dict['success']
                del tra_dict['success']
                

                if task == "PullCubeTool-v1":

                    del tra_dict['terminated']
                    del tra_dict['truncated']

                # redo the env states
                actions = tra_dict['action']
                imgs = []
                env_states = []
                obs, _ = env.reset(seed=demo_seed)
                env_states.append(env.unwrapped.get_state())
                img = env.render()  # Render the initial state
                compressed_image = np.array(Image.fromarray(img.astype('uint8')).resize(image_size))
                imgs.append(compressed_image)
                for action in actions[1:]:
                    obs, _, _, _, _ = env.step(action)

                    env_states.append(env.unwrapped.get_state())
                    img = env.render()
                    compressed_image = np.array(Image.fromarray(img.astype('uint8')).resize(image_size))
                    imgs.append(compressed_image)

                # if task == "StackCube-v0" and controller == 'pd_ee_delta_pos':

                #     extra_action = np.array([[0, 0, -0.3, 1]], dtype=np.float32)
                #     extra_steps = 10
                #     extra_obs = []

                #     for _ in range(extra_steps):
                #         obs, _, _, _, _ = env.step(extra_action[0])

                #         env_states.append(env.unwrapped.get_state())
                #         img = env.render()
                #         compressed_img = np.array(Image.fromarray(img.astype('uint8')).resize(image_size))
                #         imgs.append(compressed_img)

                #         extra_obs.append(obs)
                        

                
                #     extra_actions = np.repeat(extra_action, extra_steps, axis=0)
                #     tra_dict['action'] = np.concatenate([tra_dict['action'], extra_actions], axis=0)

                
                #     tra_dict['observation'] = np.concatenate([tra_dict['observation'], np.array(extra_obs)], axis=0)

                
                #     success = np.concatenate([success, np.array([1] * extra_steps)], axis=0)

                #     tra_dict['goal'] = np.repeat(tra_dict['observation'][-1].reshape(1, tra_dict['observation'].shape[1]), tra_dict['observation'].shape[0], axis=0)

                #     tra_dict['reward'] = np.zeros(success.shape, dtype=np.float32)

                #     tra_dict['is_first'] = np.zeros_like(success, dtype=bool)
                #     tra_dict['is_first'][0] = True

                #     tra_dict['is_last'] = np.zeros_like(success, dtype=bool)
                #     tra_dict['is_last'][-1] = True

                #     tra_dict['is_terminal'] = success

                #     tra_dict['demo_seed'] = np.zeros_like(success, dtype=int)
                #     tra_dict['demo_seed'][:] = demo_seed



                tra_dict['env_states'] = env_states
                tra_dict['images'] = np.array(imgs)
                
                
                # imageio.mimsave(f'demo_{demo_seed}.gif', imgs)
                # print("Processed trajectory for demo_seed:", demo_seed)

                all_demo_trajectories.append(tra_dict)
                find_seed_list.append(demo_seed)

        # important_features = Find_important_dim(all_demo_trajectories, if_run_all = True)

        return all_demo_trajectories, find_seed_list


    elif task == "PickAndPlace":

        if not if_random_seeds:
            if if_eval:
                seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                seed_list = [0, 1, 2, 3, 4]
                
        else:
            random.seed(1314)

            seed_list = range(random_seeds_num)

        all_demo_trajectories = []
        find_seed_list = []
        demo_all = np.load(task_demo_path, allow_pickle=True)

        # print(demo_all.keys())
        # print(demo_all['seed'])
        # print(demo_all['obs'].shape)
        # print(demo_all['act'])

        for traj_obs, traj_act, traj_seed in zip(demo_all['obs'], demo_all['act'], demo_all['seed']):

            if traj_seed not in seed_list and not if_random_seeds:
                continue

            # if traj_data['rewards'][-1] == 0:

            #     continue

            if len(find_seed_list) >= random_seeds_num:
                
                break

            tra_dict = {}

            # delete this part if load demo for training goal predictor
            # try:
            #     traj_obs = traj_obs[:, :-3]
            
            # except:
            #     for i, obs in enumerate(traj_obs):

            #         traj_obs[i] = obs[:-3]

            traj_obs = np.array(traj_obs)
            tra_dict['observation'] = traj_obs
            
            traj_act= np.array(traj_act)
            initial_action = np.zeros((1, traj_act.shape[1]))
            traj_act = np.concatenate((initial_action, traj_act), axis=0)
            tra_dict['action'] = traj_act

            tra_dict['goal'] = np.repeat(tra_dict['observation'][-1].reshape(1, tra_dict['observation'].shape[1]), tra_dict['observation'].shape[0], axis=0)
            tra_dict['reward'] = np.zeros(tra_dict['observation'].shape[0], dtype=np.float32)
            tra_dict['is_first'] = np.zeros(tra_dict['observation'].shape[0], dtype=bool)
            tra_dict['is_first'][0] = True
            tra_dict['is_last'] = np.zeros(tra_dict['observation'].shape[0], dtype=bool)
            tra_dict['is_last'][-1] = True
            tra_dict['is_terminal'] = tra_dict['is_last']
            
            all_demo_trajectories.append(tra_dict)
            find_seed_list.append(int(traj_seed))

        return all_demo_trajectories, find_seed_list


    
    elif "Adroit" in task:

        if not if_random_seeds:
            if if_eval:
                seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            else:
                seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                
        else:
            random.seed(1314)

            seed_list = range(random_seeds_num)


        import gymnasium as gym

        env = gym.make(task, render_mode="rgb_array")

        init_state_dict_keys_dict = {
            "AdroitHandDoor-v1": ['qvel', 'qpos', 'door_body_pos'],
            "AdroitHandHammer-v1": ['qvel', 'qpos', 'board_pos'],
            "AdroitHandPen-v1": ['qvel', 'qpos', 'desired_orien'],
            "AdroitHandRelocate-v1": ['qvel', 'qpos', 'obj_pos', 'target_pos']
        }

        all_demo_trajectories = []
        find_seed_list = []
        demo_all = pickle.load(open(task_demo_path, 'rb'))

        demo_idx = 0
        for demo_tra in demo_all:

            if demo_idx not in seed_list and not if_random_seeds:
                demo_idx += 1
                continue

            if len(find_seed_list) >= random_seeds_num:
                break

            init_state_dict = {key:  demo_tra['init_state_dict'][key] for key in init_state_dict_keys_dict[task]}

            tra_dict = {}
        
            tra_dict["init_state_dict"] = init_state_dict
            tra_dict["action"] = np.array(demo_tra['actions'])
            tra_dict['observation'] = []
            tra_dict['env_states'] = []
            tra_dict['images'] = []

            # print(tra_dict['observation'])

            # env states
            obs, _ = env.reset()
            env.unwrapped.set_env_state(init_state_dict)
            actions = demo_tra['actions']
            success_idx = None

            for t in range(actions.shape[0]):
                obs, reward, terminal, turncat, info = env.step(actions[t])
                # print("obs:", obs)
                tra_dict['observation'].append(obs)
                state = env.unwrapped.get_env_state()
                state = {key: state[key] for key in init_state_dict_keys_dict[task]}
                tra_dict['env_states'].append(state)

                img = env.render()
                compressed_image = np.array(Image.fromarray(img.astype('uint8')).resize(image_size))
                tra_dict['images'].append(compressed_image)

                
                if success_idx is None and info.get('success', False):
                    success_idx = t

            tra_dict['observation'] = np.array(tra_dict['observation'])
            tra_dict['env_states'] = np.array(tra_dict['env_states'])
            tra_dict['images'] = np.array(tra_dict['images'])

            tra_dict['reward'] = np.zeros(tra_dict['observation'].shape[0], dtype=np.float32)
            tra_dict['is_first'] = np.zeros(tra_dict['observation'].shape[0], dtype=bool)
            tra_dict['is_first'][0] = True
            tra_dict['is_last'] = np.zeros(tra_dict['observation'].shape[0], dtype=bool)
            tra_dict['is_last'][-1] = True
            tra_dict['is_terminal'] = tra_dict['is_last']

            # make sure the demo final obs is success
            if task == "AdroitHandDoor-v1" and tra_dict["observation"][-1][-1] != 1:
                
                success_idx = np.where(tra_dict["observation"][:, -1] == 1)[0]
                
                if len(success_idx) > 0:
                    last_success_idx = success_idx[-1]
                    
                    tra_dict["observation"] = tra_dict["observation"][:last_success_idx + 1]
                    tra_dict["action"] = tra_dict["action"][:last_success_idx + 1]
                    tra_dict["reward"] = tra_dict["reward"][:last_success_idx + 1]
                    tra_dict["is_first"] = tra_dict["is_first"][:last_success_idx + 1]
                    tra_dict["is_last"] = tra_dict["is_last"][:last_success_idx + 1]
                    tra_dict["is_terminal"] = tra_dict["is_terminal"][:last_success_idx + 1]
                    tra_dict["env_states"] = tra_dict["env_states"][:last_success_idx + 1]
                    tra_dict["images"] = tra_dict["images"][:last_success_idx + 1]
                else:
                    continue
            
            if task == "AdroitHandHammer-v1" or task == "AdroitHandPen-v1":
                if success_idx is not None:
                    if task == "AdroitHandHammer-v1":
                        tra_dict['observation'] = tra_dict['observation'][:success_idx + 1]
                        tra_dict['action'] = tra_dict['action'][:success_idx + 1]
                        tra_dict['reward'] = tra_dict['reward'][:success_idx + 1]
                        tra_dict['is_first'] = tra_dict['is_first'][:success_idx + 1]
                        tra_dict['is_last'] = tra_dict['is_last'][:success_idx + 1]
                        tra_dict['is_terminal'] = tra_dict['is_terminal'][:success_idx + 1]
                        tra_dict['env_states'] = tra_dict['env_states'][:success_idx + 1]
                        tra_dict['images'] = tra_dict['images'][:success_idx + 1]
                else:
                    print(f"Demo {demo_idx} did not reach success state.")
                    demo_idx += 1
                    continue

            all_demo_trajectories.append(tra_dict)
            find_seed_list.append(demo_idx)
            demo_idx += 1

            # images=[]
            # # obs, _ = env.reset()
            # # env.unwrapped.set_env_state(init_state_dict)
            # i = 0
            # for state in tra_dict['env_states']:
            #     i += 1
            #     # if i > 250:
            #     #     print("i:", i)
            #     # env.reset()
            #     # env.unwrapped.set_env_state(init_state_dict)
            #     env.unwrapped.set_env_state(state)
            #     images.append(env.render())

            # gif_save_path = '/common/users/yd374/ach/ACH_Server/Experiments/img/'
            # os.makedirs(gif_save_path, exist_ok=True)
            # imageio.mimsave(gif_save_path + f'demo_tra{demo_idx-1}.gif', images)
            # print("demo_tra", demo_idx-1, "saved")
            
        
        # important_features = Find_important_dim(all_demo_trajectories, env, if_run_all= True)

        return all_demo_trajectories, find_seed_list


    elif "meta" in task:

        if not if_random_seeds:
            if if_eval:
                seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                seed_list = [0, 1, 2, 3, 4]
        else:

            seed_list = range(random_seeds_num)

        all_demo_trajectories = []
        find_seed_list = []
        import torch
        data = torch.load(task_demo_path, weights_only=False)

        for traj_seed, traj_data in data.items():

            if traj_seed not in seed_list and not if_random_seeds:
                continue

            if max(traj_data['rewards']) == 0:

                continue

            if len(find_seed_list) >= random_seeds_num:
                
                break

            for key, value in traj_data.items():

                traj_data[key] = np.array(value)

            tra_dict = {}

            tra_dict['observation'] = traj_data['observations']
            tra_dict['action'] = traj_data['actions']
            tra_dict['reward'] = traj_data['rewards']

            tra_dict['goal'] = np.repeat(tra_dict['observation'][-1].reshape(1, tra_dict['observation'].shape[1]), tra_dict['observation'].shape[0], axis=0)
            tra_dict['is_first'] = np.zeros(tra_dict['observation'].shape[0], dtype=bool)
            tra_dict['is_first'][0] = True
            tra_dict['is_last'] = np.zeros(tra_dict['observation'].shape[0], dtype=bool)
            tra_dict['is_last'][-1] = True
            tra_dict['is_terminal'] = tra_dict['is_last']
            tra_dict['env_states'] = traj_data['env_states']
            
            all_demo_trajectories.append(tra_dict)

            find_seed_list.append(traj_seed)

        return all_demo_trajectories, find_seed_list


def Find_important_dim(demo_all, env=None, find_way=3, if_run_all = False):

    find_dict = {}

    if (find_way == 0 and env != None) or (if_run_all and env != None):

        print("===========================================================================================find way 0: DecisionTreeClassifier-Success")
        all_obs = []
        all_labels = []

        for demo_tra in demo_all:
            obs = np.array(demo_tra['observation'])  
            actions = np.array(demo_tra['action'])   

            obs_env, _ = env.reset()
            init_state_dict = demo_tra['init_state_dict']
            env.unwrapped.set_env_state(init_state_dict)

            success_labels = np.zeros(obs.shape[0])  

            for t in range(actions.shape[0]):  
                obs_env, reward, terminal, turncat, info = env.step(actions[t])

                if info.get('success', False):
                    success_labels[t] = 1  

            all_obs.append(obs)  
            all_labels.append(success_labels)  

        
        all_obs = np.vstack(all_obs)  # (N, obs_dim)
        all_labels = np.concatenate(all_labels)  # (N,)

        
        clf = DecisionTreeClassifier()
        clf.fit(all_obs, all_labels)

        
        y_pred = clf.predict(all_obs)
        accuracy = accuracy_score(all_labels, y_pred)
        print("Decision Tree Accuracy:", accuracy)

        
        feature_importance = clf.feature_importances_

        
        important_features = np.argsort(feature_importance)[-10:][::-1]

        print("Top 10 important feature indices:", important_features)
        print("Feature importance scores:", feature_importance[important_features])

        find_dict[0] = important_features

    if find_way == 1 or if_run_all:
        print("\n\n===========================================================================================find way 1: DecisionTreeClassifier-ProgressRegion")

        all_obs = []
        all_labels = []

        for demo_tra in demo_all:
            obs = np.array(demo_tra['observation'])
            actions = np.array(demo_tra['action'])
            
            
            obs_ratios = np.linspace(0, 1, obs.shape[0], endpoint=False)
            obs_classes = np.digitize(obs_ratios, bins=np.linspace(0, 1, 11)) - 1
            
            
            final_obs = obs[-1]
            obs_diff = obs - final_obs
            
            all_obs.append(obs_diff)
            all_labels.append(obs_classes)

        
        all_obs = np.vstack(all_obs)
        all_labels = np.concatenate(all_labels)

        
        clf = DecisionTreeClassifier()
        clf.fit(all_obs, all_labels)

        
        y_pred = clf.predict(all_obs)
        accuracy = accuracy_score(all_labels, y_pred)
        print("Decision Tree Accuracy:", accuracy)

        
        feature_importance = clf.feature_importances_

        
        # print("Feature Importance:", feature_importance)

        
        important_features = np.argsort(feature_importance)[-10:][::-1]

        print("Top 10 important feature indices:", important_features)
        print("Feature importance scores:", feature_importance[important_features])

        find_dict[1] = important_features

    if find_way == 2 or if_run_all:
        print("\n\n===========================================================================================find way 2: DecisionTreeRegressor-ProgressRatio")

        all_obs = []
        all_labels = []

        for demo_tra in demo_all:
            obs = np.array(demo_tra['observation'])
            actions = np.array(demo_tra['action'])
            
            
            obs_ratios = np.linspace(0, 1, obs.shape[0], endpoint=False)
            
            
            final_obs = obs[-1]
            obs_diff = obs - final_obs
            
            all_obs.append(obs_diff)
            all_labels.append(obs_ratios)

        
        all_obs = np.vstack(all_obs)
        all_labels = np.concatenate(all_labels)

        
        reg = DecisionTreeRegressor(max_depth=5)
        reg.fit(all_obs, all_labels)

        
        y_pred = reg.predict(all_obs)
        mse = mean_squared_error(all_labels, y_pred)
        print("Decision Tree MSE:", mse)

        
        feature_importance = reg.feature_importances_

        
        # print("Feature Importance:", feature_importance)

        
        important_features = np.argsort(feature_importance)[-10:][::-1]

        print("Top 10 important feature indices:", important_features)
        print("Feature importance scores:", feature_importance[important_features])

        find_dict[2] = important_features
        
        # save_path = "output/decision_tree.png"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)

        
        # plt.figure(figsize=(20, 10))
        # plot_tree(reg, feature_names=[f"feature_{i}" for i in range(all_obs.shape[1])], filled=True)

        
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")
        

    if find_way == 3 or if_run_all:
        print("\n\n===========================================================================================find way 3: DecisionTreeRegressor-Iteration-10best")

        all_obs = []
        all_labels = []

        
        for demo_tra in demo_all:
            obs = np.array(demo_tra['observation'])
            actions = np.array(demo_tra['action'])
            
            
            obs_ratios = np.linspace(0, 1, obs.shape[0], endpoint=False)
            
            
            final_obs = obs[-1]
            obs_diff = obs - final_obs
            
            all_obs.append(obs_diff)
            all_labels.append(obs_ratios)

        
        all_obs = np.vstack(all_obs)
        all_labels = np.concatenate(all_labels)

        selected_features = []
        available_features = list(range(all_obs.shape[1]))  

        while len(selected_features) < 10 and len(available_features) > 0:
            
            reg = DecisionTreeRegressor(
                max_depth=5,              
                min_samples_split=10,     
                min_samples_leaf=5,       
                max_leaf_nodes=50,        
                min_impurity_decrease=0.01  
            )
            reg.fit(all_obs[:, available_features], all_labels)
            
            
            feature_importance = reg.feature_importances_
            
            
            most_important = available_features[np.argmax(feature_importance)]
            selected_features.append(most_important)
            
            
            available_features.remove(most_important)
            
            print(f"Selected feature {most_important} with importance {np.max(feature_importance)}")

        print("Top 10 important feature indices:", selected_features)

        find_dict[3] = selected_features

    if find_way == 4 or if_run_all:
        print("\n\n===========================================================================================find way 4: ???")

        all_obs = []
        all_labels = []

        for demo_tra in demo_all:
            obs = np.array(demo_tra['observation'])
            num_obs = obs.shape[0]

            
            for i in range(num_obs):
                for j in range(i + 1, num_obs):
                    obs_diff = obs[i] - obs[j]  
                    label = 1 if abs(i - j) <= 1 else 0  
                    all_obs.append(obs_diff)
                    all_labels.append(label)

        
        all_obs = np.array(all_obs)
        all_labels = np.array(all_labels)

        
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(all_obs, all_labels)

        
        y_pred = clf.predict(all_obs)
        accuracy = np.mean(y_pred == all_labels)
        print("Decision Tree Accuracy:", accuracy)

        
        feature_importance = clf.feature_importances_

        
        important_features = np.argsort(feature_importance)[-10:][::-1]

        print("Top 10 important feature indices:", important_features)
        print("Feature importance scores:", feature_importance[important_features])

        find_dict[4] = important_features
        
    print("Use find_way:", find_way, "Selected features:", find_dict[find_way])


    return find_dict[find_way]



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu_2(x)
        x = self.fc3(x)

        return x


class GP_Dataset(Dataset):
    def __init__(self, demo_dataset):
        self.inputs = []
        self.targets = []
        for demo in demo_dataset:
            goal = torch.tensor(demo['observation'][-1], dtype=torch.float32)  # Goal is the last observation
            for i in range(len(demo['observation'])):
                obs = torch.tensor(demo['observation'][i], dtype=torch.float32)
                self.inputs.append(obs)
                self.targets.append(goal)
                
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class Goal_Predictor:

    def __init__(self,):
        
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent / 'Config/configs.yaml').read_text())

        parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)  

        config = common.Config(configs['defaults'])

        for name in parsed.configs:
            config = config.update(configs[name])
        config = common.Flags(config).parse(remaining)  

        self.config = config

        self.train_dataset_size = 0.7

    def load_all_demo(self,):

        demo_dataset, seed_list = get_demo_trajectories(self.config.task, self.config.demo_path, if_random_seeds=True, random_seeds_num=100, image_size=(64, 64))

        return demo_dataset, seed_list


    def train(self, model_save_path = 'mlp_model.pth'):

        demo_dataset, seed_list = self.load_all_demo()
        train_size = int(len(demo_dataset) * self.train_dataset_size)

        # Hyperparameters
        obs_size = len(demo_dataset[0]['observation'][0])
        goal_size = obs_size  # The goal is the last observation
        input_size = obs_size
        hidden_size = 100
        output_size = goal_size
        batch_size = 8
        max_steps = 100000
        learning_rate = 0.001

        # Create Dataset and DataLoader
        dataset = GP_Dataset(demo_dataset[:train_size])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss function, optimizer
        model = MLP(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop with progress bar
        step = 0
        with tqdm(total=max_steps, desc="Training Progress") as pbar:
            while step < max_steps:
                for batch_inputs, batch_targets in dataloader:

                    # Move data to GPU
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)

                    # Forward pass
                    outputs = model(batch_inputs)

                    # Compute loss
                    loss = criterion(outputs, batch_targets)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step += 1
                    pbar.update(1)
                    pbar.set_postfix({'Loss': loss.item()})
                    
                    if step >= max_steps:
                        break

        # Save your model if needed
        torch.save(model.state_dict(), model_save_path)


    # optional
    def draw_goal_img(self, goal, img_save_path = 'goal_img.png'):

        if self.config.task == 'PickAndPlace':

            env = gym.make('FetchPickAndPlace-v2', render_mode="rgb_array")
            env.reset()

            def render_with_obs(env, obs, goal, width=200, height=200):
                inner_env = env.env.env.env

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

                image = env.render()  # render the environment
            
                return image

            img = render_with_obs(env, goal[:-3], goal[-3:], width=200, height=200)

            img = Image.fromarray(img)
            img.save(img_save_path)

            env.close()


    def load_model(self, gp_model_path, obs_size, hidden_size=100):

        input_size = obs_size
        hidden_size = hidden_size
        goal_size = obs_size

        # Initialize model and load saved weights
        model = MLP(input_size, hidden_size, goal_size).to(device)
        model.load_state_dict(torch.load(gp_model_path, map_location=device))
        
        self.gp_model = model

        return model


    def predict_goal(self, obs):

        # Predict the goal based on the first observation
        input_obs = torch.tensor(obs).float().to(device).unsqueeze(0)

        self.gp_model.to(device)
        input_obs = input_obs.to(device)
        predicted_goal = self.gp_model(input_obs).detach().cpu().numpy()

        return predicted_goal
    
    
    def eval_GP(self, model_load_path='mlp_model.pth'):

        img_save_path = f'GP_MLP_Model/Images/{self.config.task}/'
        os.makedirs(img_save_path, exist_ok=True)
        # Load the model
        demo_dataset, seed_list = self.load_all_demo()
        train_size = int(len(demo_dataset) * self.train_dataset_size)
        obs_size = len(demo_dataset[0]['observation'][0])

        self.load_model(model_load_path, obs_size, hidden_size = 100)
        self.gp_model.eval()

        # Split the dataset into training and evaluation sets
        eval_dataset = demo_dataset[train_size:]
        eval_seed_list = seed_list[train_size:]

        total_error = 0.0
        num_demos = len(eval_dataset)

        # Iterate over each demo in the evaluation set
        for demo, seed in zip(eval_dataset, eval_seed_list):
            obs = demo['observation']
            true_goal = obs[-1]  # The goal is the last observation

            # Predict the goal based on the first observation
            predicted_goal = self.predict_goal(obs[0])

            # Compute the absolute error per dimension
            error = np.abs(predicted_goal - true_goal)

            # Accumulate the total error per dimension
            total_error += error

            print(f"True goal: {true_goal}")
            print(f"Predicted goal: {predicted_goal}")

            # Draw the goal image
            self.draw_goal_img(true_goal, img_save_path=img_save_path + f"Seed{seed}_true.png")
            self.draw_goal_img(predicted_goal[0], img_save_path=img_save_path + f"Seed{seed}_Predicted.png")

        # Compute the average error over all demos
        avg_error = total_error / num_demos
        print(f"Average goal prediction error: {avg_error}")
        
    
        # test seed not in demo data on PickPlace task
        # test_seed = 520
        # env = gym.make('FetchPickAndPlace-v2', render_mode="rgb_array")
        # obs = env.reset(seed = test_seed)
        # initial_obs = np.concatenate((obs[0]['observation'], obs[0]['desired_goal']))
        # predicted_goal = self.predict_goal(initial_obs)

        # self.draw_goal_img(initial_obs, task=task, img_save_path=img_save_path + f"Seed{test_seed}_initial.png")
        # self.draw_goal_img(predicted_goal[0], task=task, img_save_path=img_save_path + f"Seed{test_seed}_Predicted.png")


    def main(self,):

        # task = 'PickPlace'
        # task = 'PegInsertation'
        model_save_path = f'GP_MLP_Model/mlp_model_{self.config.task}.pth'

        self.train(model_save_path = model_save_path)
        self.eval_GP(model_load_path = model_save_path)


# ============================================================================================================================

class CNNNet(nn.Module):
    def __init__(self, input_height, input_width):
        super(CNNNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # -> [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # -> [B, 128, H/8, W/8]
            nn.ReLU(),
        )

        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            encoded = self.encoder(dummy)
            self.flatten_shape = encoded.shape[1:]         # (C, H, W)
            self.flatten_dim = encoded.numel()             # C*H*W

        self.encoder_flatten = nn.Flatten()

        self.decoder = nn.Sequential(
            nn.Linear(self.flatten_dim, self.flatten_dim),
            nn.ReLU(),
            nn.Unflatten(1, self.flatten_shape),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_flatten(x)
        x = self.decoder(x)
        return x


class ImageGP_Dataset(Dataset):
    def __init__(self, demo_dataset, image_size=(64, 64)):
        self.inputs = []
        self.targets = []
        for demo in demo_dataset:
            goal = demo['images'][-1]
            goal_img = self.preprocess(goal, image_size)
            for img in demo['images']:
                obs_img = self.preprocess(img, image_size)
                self.inputs.append(obs_img)
                self.targets.append(goal_img)

    def preprocess(self, img, image_size):
        img = Image.fromarray(img)
        img = img.resize(image_size)
        img = np.asarray(img).astype(np.float32)  # No normalization
        img = torch.tensor(img).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class Image_Goal_Predictor:
    def __init__(self):
        
        configs = yaml.safe_load(
            (pathlib.Path(sys.argv[0]).parent.parent / 'Config/configs.yaml').read_text())
        parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
        config = common.Config(configs['defaults'])

        for name in parsed.configs:
            config = config.update(configs[name])
        config = common.Flags(config).parse(remaining)

        self.config = config
        self.train_dataset_size = 0.7
        self.image_size = (64, 64)

        
        self.model = CNNNet(input_height=self.image_size[0], input_width=self.image_size[1]).to(device)

    def load_all_demo(self):
        demo_dataset, seed_list = get_demo_trajectories(
            self.config.task,
            self.config.demo_path,
            if_random_seeds=True,
            random_seeds_num=30,
            image_size=self.image_size
        )
        return demo_dataset, seed_list

    def train(self, model_save_path='cnn_model.pth', batch_size=16, lr=1e-3, max_steps=5000):
        demo_dataset, seed_list = self.load_all_demo()
        train_size = int(len(demo_dataset) * self.train_dataset_size)
        train_dataset = ImageGP_Dataset(demo_dataset[:train_size], self.image_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        step = 0
        with tqdm(total=max_steps, desc="Training CNN") as pbar:
            while step < max_steps:
                for inputs, targets in train_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    step += 1
                    pbar.update(1)
                    pbar.set_postfix({'Loss': loss.item()})

                    if step >= max_steps:
                        break

        torch.save(self.model.state_dict(), model_save_path)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict_goal(self, image):
        img = Image.fromarray(image).resize(self.image_size)
        img = np.asarray(img).astype(np.float32)
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = self.model(img_tensor).squeeze(0).cpu().numpy()
        
        predicted = np.clip(predicted, 0, 255).astype(np.uint8)  # Clip to valid range
        return predicted


    def draw_goal_img(self, goal, img_save_path='goal_img.png'):
        import imageio.v2 as imageio  
        imageio.imwrite(img_save_path, goal.astype(np.uint8))

    def eval_GP(self, model_load_path='cnn_model.pth'):
        img_save_path = f'GP_CNN_Model/Images/{self.config.task}/'
        os.makedirs(img_save_path, exist_ok=True)

        demo_dataset, seed_list = self.load_all_demo()
        train_size = int(len(demo_dataset) * self.train_dataset_size)
        eval_dataset = demo_dataset[train_size:]
        eval_seed_list = seed_list[train_size:]

        train_dataset = demo_dataset[:train_size]
        train_seed_list = seed_list[:train_size]

        self.load_model(model_load_path)

        print("Evaluate the train seeds...")
        train_seeds_img_save_path = img_save_path + "Train_seeds/"
        os.makedirs(train_seeds_img_save_path, exist_ok=True)
        for demo, seed in zip(train_dataset, train_seed_list):
            images = demo['images']
            true_goal = demo['images'][-1]

            predicted_goal = self.predict_goal(images[0])
            predicted_goal = np.transpose(predicted_goal, (1, 2, 0))

            # print(f"True goal: {true_goal}")
            # print(f"Predicted goal: {predicted_goal}")

            self.draw_goal_img(true_goal, train_seeds_img_save_path + f"Seed{seed}_true.png")
            self.draw_goal_img(predicted_goal, train_seeds_img_save_path + f"Seed{seed}_predicted.png")



        print("Evaluate the eval seeds...")
        eval_seeds_img_save_path = img_save_path + "Eval_seeds/"
        os.makedirs(eval_seeds_img_save_path, exist_ok=True)
        for demo, seed in zip(eval_dataset, eval_seed_list):
            images = demo['images']
            true_goal = demo['images'][-1]

            predicted_goal = self.predict_goal(images[0])  
            predicted_goal  = np.transpose(predicted_goal, (1, 2, 0))

            # print(f"True goal: {true_goal}")
            # print(f"Predicted goal: {predicted_goal}")

            self.draw_goal_img(images[0], eval_seeds_img_save_path + f"Seed{seed}_initial.png")
            self.draw_goal_img(true_goal, eval_seeds_img_save_path + f"Seed{seed}_true.png")
            self.draw_goal_img(predicted_goal, eval_seeds_img_save_path + f"Seed{seed}_predicted.png")



    def main(self):
        model_save_path = f'GP_CNN_Model/cnn_model_{self.config.task}.pth'

        print("Training CNN model...")
        self.train(model_save_path=model_save_path)
        print("Evaluating CNN model...")
        self.eval_GP(model_load_path=model_save_path)

    


if __name__ == '__main__':

    # state observation
    # GP = Goal_Predictor()
    # GP.main()

    # image observation
    GP = Image_Goal_Predictor()
    GP.main()

# python dreamerv2_demo/Goal_Predictor.py --configs AdroitDoor














