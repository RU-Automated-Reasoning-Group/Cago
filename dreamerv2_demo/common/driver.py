
import numpy as np

import itertools
import random
import copy
import tensorflow as tf
import h5py
import time


class Driver:

    def __init__(self, envs, **kwargs):
        self._envs = envs
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)


    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):

        step, episode = 0, 0

        while step < steps or episode < episodes:

            obs = {i: self._envs[i].reset() for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}

            for i, ob in obs.items():

                self._obs[i] = ob() if callable(ob) else ob

                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}

                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]

                self._eps[i] = [tran]


            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}


            obs = {key: value for key, value in obs.items() if key not in ['env_states']} # important to remove env_states for policy.
            actions, self._state = policy(obs, self._state, **self._kwargs)


            actions = [{k: np.array(actions[k][i]) for k in actions} for i in range(len(self._envs))]

            assert len(actions) == len(self._envs)

            obs = [e.step(a) for e, a in zip(self._envs, actions)]


            obs = [ob() if callable(ob) else ob for ob in obs]


            for i, (act, ob) in enumerate(zip(actions, obs)):

                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]

                self._eps[i].append(tran)

                step += 1

                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1

            self._obs = obs


    def _convert(self, value):


        value = np.array(value)

        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        
        return value


class GCDriver(Driver):
    def __init__(self, envs, goal_key, config, **kwargs):
        super().__init__(envs, **kwargs)
        self.config = config
        self.goal_key = goal_key
        self.all_transfer_goals = None

        self.all_3_block_train_goals_index = [10, 11, 14]

        self.if_eval_driver = False

        self.if_set_initial_state = False
        self.set_state_fn = None
        self.initial_state = None


    def reset(self):
        super().reset()
        self._subgoals = [None] * len(self._envs)
        self._use_policy_2 = [False] * len(self._envs)
        self._goal_time = [0] * len(self._envs)
        self._goal_dist = [0] * len(self._envs) # store subgoal dist per episode.
        self._goal_success = [0] * len(self._envs) # store subgoal success per episode.


    def __call__(self, policy_1, 
                 policy_2=None, 
                 get_goal=None, 
                 goal_optimizer=None,
                 steps=0, 
                 episodes=0, 
                 goal_time_limit=None, 
                 goal_checker=None, 
                 if_multi_3_blcok_training_goal = False, 
                 label = 'Normal'):
        
        """
        1. train: run gcp for entire rollout using goals from buffer/search.
        2. expl: run plan2expl for entire rollout
        3. 2pol: run gcp with goals from buffer/search and then expl policy
        
        LEXA is (1,2) and choosing goals from buffer.
        Ours can be (1,2,3), or (1,3) and choosing goals from search
        
        Args:
                policy_1 (_type_): 1st policy to run in episode
                policy_2 (_type_, optional): 2nd policy that runs after first policy is done. If None, then only run 1st policy.
                goal_strategy (_type_, optional): How to sample a goal
                steps (int, optional): _description_. Defaults to 0.
                episodes (int, optional): _description_. Defaults to 0.
                goal_time_limit (_type_, optional): _description_. Defaults to None.
                goal_checker (_type_, optional): _description_. Defaults to None.
        """

        # s = 0
        # time_record = time.time()
        # print(s)
        step, episode = 0, 0
        while step < steps or episode < episodes:

            obs = {}
            for i, ob in enumerate(self._obs):

                if ob is None or ob['is_last']:

                    if if_multi_3_blcok_training_goal:

                        self.training_goal_index = random.randint(1, 3)
                        training_env_goal_index = self.all_3_block_train_goals_index[self.training_goal_index-1]

                        label = 'egc' + str(self.training_goal_index)
                        
                        self._envs[i].set_goal_idx(training_env_goal_index)

                    obs[i] = self._envs[i].reset()

                    if self.if_eval_driver and self.if_set_initial_state:
                         
                        self.set_state_fn(self._envs[i], self.initial_state)
                        obs[i] = self._envs[i].step({'action': np.zeros(self._act_spaces[i]['action'].shape)})

                        self.if_set_initial_state = False      

            # initialize
            for i, ob in obs.items():

                if self.config.if_goal_optimizer:
                    if goal_optimizer is not None:
                        ob_copy = {key: np.stack([value]) for key, value in ob.items() if key not in ['env_states']}
                        optimized_goal = goal_optimizer.give_optimized_goal(ob_copy)
                        ob['optimized_goal'] = optimized_goal[0].numpy()
                    else:
                        ob['optimized_goal'] = ob[self.goal_key]

                # Same with Driver
                self._obs[i] = ob() if callable(ob) else ob

                # Same with Driver
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}

                # Same with Driver
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}

                self._use_policy_2[i] = False
                self._goal_time[i] = 0


                if get_goal:
                    
                    if self.config.if_image_obs:
                        sub_state_goal, image_goal = get_goal(obs, self._state, env=self._envs[i])
                        subgoal = image_goal
                        self._subgoals[i] = image_goal
                        tran[self.goal_key] = image_goal.numpy()
                        tran['goal'] = sub_state_goal.numpy()

                    
                    else:
                        subgoal = get_goal(obs, self._state, env=self._envs[i]) 
                        self._subgoals[i] = subgoal
                        tran[self.goal_key] = subgoal.numpy()
                

                tran["label"] = label
                # Same with Driver
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]

                if goal_checker is not None:
                    # _, goal_info = goal_checker(obs) # update goal distance metric
                    self._goal_dist[i] = 0
                    self._goal_success[i] = 0.0

                # Same with Driver
                self._eps[i] = [tran]

            obs = {}

            for k in self._obs[0]:
                if k == self.goal_key: # use subgoal if generated else use original goal.
                    goals = [g if g is not None and get_goal else self._obs[i][k] for (i,g) in enumerate(self._subgoals)]
                    obs[k] = np.stack(goals)
                    if self.config.if_image_obs and get_goal:
                        obs['goal'] = np.stack([sub_state_goal])
                else:
                    obs[k] = np.stack([o[k] for o in self._obs])
            

            policy = policy_2 if self._use_policy_2[0] else policy_1

            obs = {key: value for key, value in obs.items() if key not in ['env_states']}  # important to remove env_states for policy.

            if self.config.if_goal_optimizer:
                if goal_optimizer is not None and 'optimized_goal' in obs.keys():                 
                    original_goal = obs[self.goal_key]
                    optimized_goal = obs['optimized_goal']
                    obs[self.goal_key] = optimized_goal
                    actions, self._state = policy(obs, self._state, **self._kwargs)
                    obs[self.goal_key] = original_goal
                    obs['optimized_goal'] = optimized_goal
                else:
                    actions, self._state = policy(obs, self._state, **self._kwargs)
            else:
                actions, self._state = policy(obs, self._state, **self._kwargs)
                # except Exception as e:
            #     print("Error in policy call, error: ", e)
            #     print(policy, self._use_policy_2[0], policy_2)

            actions = [{k: np.array(actions[k][i]) for k in actions} for i in range(len(self._envs))]

            assert len(actions) == len(self._envs)

            obs = [e.step(a) for e, a in zip(self._envs, actions)]
            obs = [ob() if callable(ob) else ob for ob in obs]

            
            if self.config.if_goal_optimizer:

                for i, ob in enumerate(obs):
                    if goal_optimizer is not None:
                        ob_copy = {key: np.stack([value]) for key, value in ob.items() if key not in ['env_states']}
                        optimized_goal = goal_optimizer.give_optimized_goal(ob_copy)
                        ob['optimized_goal'] = optimized_goal[0].numpy()
                    else:
                        ob['optimized_goal'] = ob[self.goal_key]

            if get_goal: # overwrite goal since obs just came from env.
                for o in obs:
                    o[self.goal_key] = subgoal.numpy()
                    if self.config.if_image_obs:
                        o['goal'] = sub_state_goal.numpy()

            # now check if obs achieved subgoal or not.
            for i, ob in enumerate(obs):

                if policy_2 is None or self._use_policy_2[i]:
                    continue

                self._goal_time[i] += 1
                subgoal = self._subgoals[i]
                out_of_time = goal_time_limit and self._goal_time[i] > goal_time_limit

                if self.config.if_actor_gs:

                    close_to_goal, goal_info = False, {}

                else:
                    close_to_goal, goal_info = goal_checker(ob)
                    self._goal_dist[i] += goal_info["subgoal_dist"]
                    self._goal_success[i] += goal_info["subgoal_success"]

                if out_of_time or close_to_goal:
                    self._use_policy_2[i] = True

            # Same with Driver
            for i, (act, ob) in enumerate(zip(actions, obs)):
                # print(ob['observation'])
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                # print(tran['observation'])
                tran["label"] = label

                # print(s,'before_train', time.time()-time_record)

                # time_record = time.time()
                
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                # print(s, 'after_train', time.time()-time_record)
                # time_record = time.time()
                # s += 1
                step += 1

                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}

                    ep["log_subgoal_dist"] = np.array([self._goal_dist[i]]) # add subgoal metrics
                    ep["log_subgoal_success"] = np.array([float(self._goal_success[i] > 0)]) # add subgoal metrics
                    ep["log_subgoal_time"] = np.array([self._goal_time[i]]) # time to reach subgoal.

                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1

            self._obs = obs







