

from collections import defaultdict
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from time import time
import random
import heapq
import h5py
import copy
import tensorflow_probability as tfp


class GC_goal_picker:

    def __init__(self, config, agnt, replay, dataset, env, obs2goal_fn, sample_env_goals_fn, vis_fn):
        
        

        if config.goal_strategy == "Greedy":
            goal_strategy = Greedy(replay, agnt.wm, agnt._expl_behavior._intr_reward, config.state_key, config.goal_key, 1000)

        elif config.goal_strategy == "SampleReplay":
            goal_strategy = SampleReplay(agnt.wm, dataset, config.state_key, config.goal_key)

        elif config.goal_strategy == "MEGA":
            goal_strategy = MEGA(config, agnt, replay, env.act_space, config.state_key, config.time_limit, obs2goal_fn)

        elif config.goal_strategy == "Skewfit":
            goal_strategy = Skewfit(agnt, replay, env.act_space, config.state_key, config.time_limit, obs2goal_fn)

        # PEG
        elif config.goal_strategy == "SubgoalPlanner":

            goal_strategy = SubgoalPlanner(
                agnt,
                config,
                env,
                replay,
                obs2goal_fn=obs2goal_fn,
                sample_env_goals_fn = sample_env_goals_fn,
                vis_fn=vis_fn
            )

        # Cluster
        elif config.goal_strategy == "Cluster_goal_Planner":

            goal_strategy = Cluster_goal_Planner(
                agnt,
                config,
                env,
                obs2goal_fn=obs2goal_fn,
            )

        # Demo goal
        elif config.goal_strategy == "Demo_goal_Planner":

            goal_strategy = Demo_goal_Planner(
                agnt,
                config,
                env,
                obs2goal_fn=obs2goal_fn,
            )

        else:
            raise NotImplementedError
        
        self.goal_strategy = goal_strategy

        self.get_goal_fn = self.make_get_goal_fn(config, agnt, sample_env_goals_fn)


    def make_get_goal_fn(self, config, agnt, sample_env_goals_fn):

        
        def get_goal(obs, state=None, mode='train', env = None, **kwargs):
            
            obs = tf.nest.map_structure(lambda x: tf.expand_dims(
                tf.expand_dims(tf.tensor(x), 0), 0), obs)[0]
            obs = {key: value for key, value in obs.items() if key not in ['env_states']}
            obs = agnt.wm.preprocess(obs)
            if np.random.uniform() < config.planner.sample_env_goal_percent:
                goal = sample_env_goals_fn(1)
                return tf.squeeze(goal)

            if config.goal_strategy == "Greedy":
                goal = self.goal_strategy.get_goal(**kwargs)
                self.goal_strategy.will_update_next_call = False
            elif config.goal_strategy == "SampleReplay":
                goal = self.goal_strategy.get_goal(obs, **kwargs)
            elif config.goal_strategy in {"MEGA", "Skewfit"}:
                goal = self.goal_strategy.sample_goal(obs, state, **kwargs)
            elif config.goal_strategy == "SubgoalPlanner":
                goal = self.goal_strategy.search_goal(obs, state, **kwargs)
                self.goal_strategy.will_update_next_call = False
            elif config.goal_strategy == "Cluster_goal_Planner":
                goal = self.goal_strategy.search_goal(obs, state, **kwargs)
            elif config.goal_strategy == "Demo_goal_Planner":

                if config.if_image_obs:
                    goal, image_goal = self.goal_strategy.search_goal(env=env, **kwargs)

                    return tf.squeeze(goal) , tf.squeeze(image_goal)
                else:
                    goal = self.goal_strategy.search_goal(env=env, **kwargs)
            else:
                raise NotImplementedError
            return tf.squeeze(goal)
        
        return get_goal



class Greedy:
    def __init__(self, replay, wm, reward_fn, state_key, goal_key, batch_size, topk=10, exp_weight=1.0):
        self.replay = replay
        self.wm = wm
        self.reward_fn = reward_fn
        self.state_key = state_key
        self.goal_key = goal_key
        self.batch_size = batch_size
        self.topk = topk
        self.exp_weight = exp_weight
        self.all_topk_states = None

    def update_buffer_priorities(self):
        start = time()

        # go through the entire replay buffer and extract top K goals.
        @tf.function
        def process_batch(data, reward_fn):
            data = self.wm.preprocess(data)
            states = data[self.state_key]
            # need to pass states through encoder / rssm first to get 'feat'
            embed = self.wm.encoder(data)
            post, prior = self.wm.rssm.observe(
                    embed, data['action'], data['is_first'], state=None)
            data['feat'] = self.wm.rssm.get_feat(post)
            # feed these states into the plan2expl loss.
            reward = reward_fn(data).reshape((-1,))
            values, indices = tf.math.top_k(reward, self.topk)
            states = data[self.state_key].reshape((-1, data[self.state_key].shape[-1]))
            topk_states = tf.gather(states, indices)
            # last_state = {k: v[:, -1] for k, v in post.items()}
            return values, topk_states
        
        self.all_topk_states = []
        # reward_fn = agent._expl_behavior._intr_reward
        # this dict contains keys (file paths) and values (episode dicts)
        num_episodes = len(self.replay._complete_eps)
        chunk = defaultdict(list)
        count = 0
        for idx, ep_dict in enumerate(self.replay._complete_eps.values()):
            for k,v in ep_dict.items():
                chunk[k].append(v)
            count += 1
            if count >= self.batch_size or idx == num_episodes-1: # done with collecting chunk.
                count = 0
                data = {k: np.stack(v) for k,v in chunk.items()}
                # for k, v in data.items():
                #     print(k, v.shape)
                chunk = defaultdict(list)
                # do processing of batch here.
                values, top_states = process_batch(data, self.reward_fn)
                values_states = [(v,s) for v,s in zip(values, top_states)]
                self.all_topk_states.extend(values_states)
                self.all_topk_states.sort(key=lambda x: x[0], reverse=True)
                self.all_topk_states = self.all_topk_states[:self.topk]
        end = time() - start
        print("update buffer took", end)


    def get_goal(self):
        if self.all_topk_states is None:
            self.update_buffer_priorities()

        priorities = np.asarray([x[0] for x in self.all_topk_states])
        priorities += 1e-6    # epsilon to prevent collapse
        np.exp(priorities * self.exp_weight)
        prob = np.squeeze(priorities) / priorities.sum()

        
        idx = np.random.choice(len(self.all_topk_states), 1, replace=True, p=prob)[0]
        value, state = self.all_topk_states[idx]
        return state.numpy()



class SampleReplay:
    def __init__(self, wm, dataset, state_key, goal_key):
        self.state_key = state_key
        self.goal_key = goal_key
        self._dataset = dataset
        self.wm = wm

    @tf.function
    def get_goal(self, obs, **kwargs):
        random_batch = next(self._dataset)
        random_batch = self.wm.preprocess(random_batch)
        random_goals = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
        return random_goals[:obs[self.state_key].shape[0]]


class MEGA:
    def __init__(self, config, agnt, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn=None):

         # agnt things
        self.agnt = agnt
        self.config = config
        self.wm = agnt.wm
        self.obs2goal_fn = obs2goal_fn
        self.dtype = agnt.wm.dtype
        if self.config.if_actor_gs:
            self.actor = agnt._task_behavior.actor_gs
        else:
            self.actor = agnt._task_behavior.actor
        

        if self.config.expl_behavior == 'Plan2Explore':
            self.reward_fn = agnt._expl_behavior.planner_intr_reward

        # p_cfg
        p_cfg = config.planner
        self.planner = p_cfg.planner_type
        self.horizon = p_cfg.horizon
        self.batch = p_cfg.batch
        self.cem_elite_ratio = p_cfg.cem_elite_ratio
        self.optimization_steps = p_cfg.optimization_steps
        self.std_scale = p_cfg.std_scale
        self.mppi_gamma = p_cfg.mppi_gamma
        self.evaluate_only = p_cfg.evaluate_only  
        self.repeat_samples = p_cfg.repeat_samples  
        self.env_goals_percentage = p_cfg.init_env_goal_percent
        
        # env
        self.act_space = act_space
        if isinstance(self.act_space, dict):
            self.act_space = self.act_space['action']

        # config
        self.gc_input = config.gc_input
        self.state_key = config.state_key

        
        state = None
        if state is None:
            self.initial_latent = self.wm.rssm.initial(1)
            self.initial_action = tf.zeros((1,) + self.act_space.shape)

        self.decoder = self.wm.heads['decoder']


        self.replay = replay
        self.goal_sample_fn = goal_sample_fn

        # TODO: remove hardcoding
        self.dataset = iter(replay.dataset(batch=10, length=ep_length))
        ## KDE STUFF
        from sklearn.neighbors import KernelDensity
        self.alpha = -1.0
        self.kernel = 'gaussian'
        self.bandwidth = 0.1
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kde_sample_mean = 0.
        self.kde_sample_std = 1.

        self.state_key = state_key
        self.ready = False
        self.random = False
        self.ep_length = ep_length
        self.obs2goal = obs2goal_fn

        # ACH Add： MEGA + PEG
        self.if_exploration_potential_filter = True
        self.if_eval_fitness = self.if_exploration_potential_filter
        self.elite_num = 10

    def update_kde(self):
        self.ready = True
        # Follow RawKernelDensity in density.py of MRL codebase.

        # ========== Sample Goals=============
        # we know ep length is 51
        num_episodes = self.replay.stats['loaded_episodes']
        # sample 10K goals from the buffer.
        num_samples = min(10000, self.replay.stats['loaded_steps'])
        # first uniformly sample from episodes.
        ep_idx = np.random.randint(0, num_episodes, num_samples)
        # uniformly sample from timesteps
        t_idx = np.random.randint(0, self.ep_length, num_samples)
        # store all these goals.
        all_episodes = list(self.replay._complete_eps.values())
        if self.obs2goal is None:
            kde_samples = [all_episodes[e][self.state_key][t] for e,t in zip(ep_idx, t_idx)]
        else:
            kde_samples = [self.obs2goal(all_episodes[e][self.state_key][t]) for e,t in zip(ep_idx, t_idx)]
        # normalize goals
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std
    #     # Now also log the entropy
    #     if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
    #         # Scoring samples is a bit expensive, so just use 1000 points
    #         num_samples = 1000
    #         s = self.fitted_kde.sample(num_samples)
    #         entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
    #         self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)


        # =========== Fit KDE ================
        self.fitted_kde = self.kde.fit(kde_samples)

    def evaluate_log_density(self, samples):
        # print(samples.shape)
        assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
        return self.fitted_kde.score_samples( (samples    - self.kde_sample_mean) / self.kde_sample_std )

    def sample_goal(self, obs, state=None, mode='train', **kwargs):
        if not self.ready:
            self.update_kde()
        if self.goal_sample_fn:
            num_samples = 10000
            sampled_ags = self.goal_sample_fn(num_samples)
        else:
            # ============ Sample goals from buffer ============
            # random_batch = next(self.dataset)
            # random_batch = self.wm.preprocess(random_batch)
            # sampled_ags = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
            num_episodes = self.replay.stats['loaded_episodes']
            # sample 10K goals from the buffer.
            num_samples = min(10000, self.replay.stats['loaded_steps'])
            # first uniformly sample from episodes.
            ep_idx = np.random.randint(0, num_episodes, num_samples)
            # uniformly sample from timesteps
            t_idx = np.random.randint(0, self.ep_length, num_samples)
            # store all these goals.
            all_episodes = list(self.replay._complete_eps.values())

            
            if self.obs2goal is None:
                sampled_ags = np.asarray([all_episodes[e][self.state_key][t] for e,t in zip(ep_idx, t_idx)])
            else:
                sampled_ags = np.asarray([self.obs2goal(all_episodes[e][self.state_key][t]) for e,t in zip(ep_idx, t_idx)])

            # if self.obs2goal is not None:
            #     sampled_ags = self.obs2goal(sampled_ags)

            # print(sampled_ags.shape)
        # ============Q cutoff ================
        # NOT IMPORTANT FOR MAZE, so ignore.
        # # 1. get feat of state.
        # if state is None:
        #     latent = self.wm.rssm.initial(1)
        #     action = tf.zeros((1,1,) + self.act_space.shape)
        #     state = latent, action
        # else:
        #     latent, action = state
        # embed = self.wm.encoder(obs)
        # post, prior = self.wm.rssm.observe( # q(s' | e,a,s)
        #         embed, action, obs['is_first'], latent)
        # start_state = {k: v[:, -1] for k, v in post.items()}
        # start_state['feat'] = self.wm.rssm.get_feat(start_state) # (1, 1800)

        # start = tf.nest.map_structure(lambda x: tf.repeat(x, sampled_ags.shape[0],0), start_state)
        # goal_obs = start.copy()
        # goal_obs[self.state_key] = sampled_ags
        # goal_embed = self.wm.encoder(goal_obs)
        # feat_goal = tf.concat([start['feat'], goal_embed], -1)
        # q_values = self.agent._task_behavior.critic(feat_goal).mode()
        # import ipdb; ipdb.set_trace()
        # bad_q_idxs = q_values < self.cutoff
        q_values = None
        bad_q_idxs = None

        # ============ Scoring ==================
        sampled_ag_scores = self.evaluate_log_density(sampled_ags)
        # Take softmax of the alpha * log density.
        # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
        # If alpha < -1, this skews the density to give us low density samples
        normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha)
        normalized_inverse_densities *= -1.    # make negative / reverse order so that lower is better.
        goal_values = normalized_inverse_densities
        # ============ Get Minimum Density Goals ===========
        if q_values is not None:
            goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

        if self.random:
            abs_goal_values = np.abs(goal_values)
            normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=0, keepdims=True)
            # chosen_idx = (normalized_values.cumsum(0) > np.random.rand(normalized_values.shape[0])).argmax(0)
            chosen_idx = np.random.choice(len(abs_goal_values), 1, replace=True, p=normalized_values)[0]
        else:

            if self.if_exploration_potential_filter:

                
                if state is None:
                    latent = self.wm.rssm.initial(1)
                    action = tf.zeros((1, 1,) + self.act_space.shape)
                    state = latent, action
                    # print("make new state")
                else:
                    latent, action = state
                    action = tf.expand_dims(action, 0)
                    # action should be (1, 1, D)
                    # print("using exisitng state")

                
                # create start state.
                embed = self.wm.encoder(obs)
                # posterior is q(s' | s,a,e)
                post, prior = self.wm.rssm.observe(embed, action, obs['is_first'], latent)  # post: {''stoch': z_t+1, 'deter': h_t+1}
                # for k,v in post.items():
                
                init_start = {k: v[:, -1] for k, v in post.items()}  # (1, 1, d) --> (1, d)
                
                # for k,v in latent.items():
                
                
                
                @tf.function
                def eval_fitness(goal):

                    start = {k: v for k, v in init_start.items()}
                    start['feat'] = self.wm.rssm.get_feat(start)  

                    
                    start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)


                    if self.gc_input == "state" or self.config.if_actor_gs:
                        goal_input = tf.cast(goal, self.dtype)  
                    
                    elif self.gc_input == "embed":
                        goal_obs = start.copy()
                        goal_obs[self.state_key] = goal
                        goal_input = self.wm.encoder(goal_obs)  

                    actor_inp = tf.concat([start['feat'], goal_input], -1)  

                    start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
                    seq = {k: [v] for k, v in start.items()}  

                    
                    for _ in range(self.horizon):
                        actor_inp = tf.concat([seq['feat'][-1], goal_input], -1)
                        action = self.actor(actor_inp).sample()
                        state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
                        feat = self.wm.rssm.get_feat(state)
                        for key, value in {**state, 'action': action, 'feat': feat}.items():
                            seq[key].append(value)

                    seq = {k: tf.stack(v, 0) for k, v in seq.items()}  
                    # rewards should be (batch,1)
                    rewards = self.reward_fn(seq)  

                    # for k,v in seq.items():
                    
                    

                    returns = tf.reduce_sum(rewards, 0)  

                    
                    # rewards = tf.ones([goal.shape[0],])
                    return returns, seq

                min_n_indices = heapq.nsmallest(self.elite_num, range(len(goal_values)), key=goal_values.__getitem__)

                elites = [sampled_ags[i] for i in min_n_indices]
                elites = np.array(elites)

                samples = tf.convert_to_tensor(elites, dtype=self.dtype)

              
                if self.config.expl_behavior == 'Plan2Explore' and self.if_eval_fitness:
                    
                    fitness, seq = eval_fitness(samples)

                    weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)

                    max_indices = int(tf.argmax(weights).numpy())
                    # print(max_indices)
                    # print(weights[max_indices])
                    explore_goal = samples[max_indices]
                    # print(weights)
                    # print(weights.shape)

                else:
                    explore_goal = random.choice(samples)

            else:

                chosen_idx = np.argmin(goal_values)

                explore_goal = sampled_ags[chosen_idx]


        # Store if we need the MEGA goal distribution
        self.sampled_ags = sampled_ags
        self.goal_values = goal_values

        return explore_goal


class Skewfit(MEGA):
    def __init__(self, agent, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn):
        super().__init__(agent, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn)
        self.random = True


# PEG


class SubgoalPlanner:

    def __init__(
            self, 
            agnt, 
            config, 
            env, 
            replay, 
            obs2goal_fn = None, 
            sample_env_goals_fn=None,
            vis_fn=None,):

        # agnt things
        self.wm = agnt.wm
        self.config = config
        self.dtype = agnt.wm.dtype
        if self.config.if_actor_gs:
            self.actor = agnt._task_behavior.actor_gs
        else:
            self.actor = agnt._task_behavior.actor
        self.reward_fn = agnt._expl_behavior.planner_intr_reward

        # p_cfg
        p_cfg = config.planner
        shape = env.obs_space[config.state_key].shape

        
        self.min_goal = np.full(shape, -np.inf, dtype=np.float32)
        self.max_goal = np.full(shape, np.inf, dtype=np.float32)
        self.planner = p_cfg.planner_type
        self.horizon = p_cfg.horizon
        self.batch = p_cfg.batch
        self.cem_elite_ratio = p_cfg.cem_elite_ratio
        self.optimization_steps = p_cfg.optimization_steps
        self.std_scale = p_cfg.std_scale
        self.mppi_gamma = p_cfg.mppi_gamma
        self.evaluate_only = p_cfg.evaluate_only  
        self.repeat_samples = p_cfg.repeat_samples  
        self.env_goals_percentage = p_cfg.init_env_goal_percent
        
        # env
        self.goal_dim = np.prod(env.obs_space[config.goal_key].shape)
        self.act_space = env.act_space
        if isinstance(self.act_space, dict):
            self.act_space = self.act_space['action']
        self.obs2goal = obs2goal_fn
        self.sample_env_goals = self.env_goals_percentage > 0
        self.sample_env_goals_fn = sample_env_goals_fn if self.sample_env_goals else None

        # config
        self.gc_input = config.gc_input
        self.state_key = config.state_key

        self.vis_fn = vis_fn
        self.will_update_next_call = True

        # initialize candidates
        # ugly hack for specifying no init cand.
        if p_cfg.init_candidates[0] == 123456789.0:
            init_cand = None
        else:
            init_cand = np.array(p_cfg.init_candidates, dtype=np.float32)
            # unflatten list of init candidates
            # assume goal dim = state dim
            goal_dim = np.prod(env.obs_space[config.state_key].shape)
            assert len(init_cand) == goal_dim, f"{len(init_cand)}, {goal_dim}"
            init_cand = np.split(init_cand, len(init_cand)//goal_dim)
            init_cand = tf.convert_to_tensor(init_cand)

        self.init_distribution = None
        if init_cand is not None:
            self.create_init_distribution(init_cand)

        # goal dataset
        goal_dataset = None

        
        if p_cfg.sample_replay:
            # take 10K states.
            goal_dataset = iter(replay.dataset(batch=10000//(config.time_limit+1), length=config.time_limit+1))

        self.dataset = goal_dataset
        if self.evaluate_only:
            assert self.dataset is not None, "need to sample from replay buffer."


    
    def search_goal(self, obs, state=None, **kwargs):

        # print("Goal search")
        
        

        
        if self.will_update_next_call is False:
            return self.sample_goal()

        elite_size = int(self.batch * self.cem_elite_ratio)

        
        if state is None:
            latent = self.wm.rssm.initial(1)
            action = tf.zeros((1,1,) + self.act_space.shape)
            state = latent, action
            # print("make new state")
        else:
            latent, action = state
            action = tf.expand_dims(action, 0)
            # action should be (1, 1, D)
            # print("using exisitng state")

        
        # create start state.
        embed = self.wm.encoder(obs)
        # posterior is q(s' | s,a,e)
        post, prior = self.wm.rssm.observe(embed, action, obs['is_first'], latent)  # post: {''stoch': z_t+1, 'deter': h_t+1}
        # for k,v in post.items():
        
        init_start = {k: v[:, -1] for k, v in post.items()}  # (1, 1, d) --> (1, d)
        
        # for k,v in latent.items():
        

        
        @tf.function
        def eval_fitness(goal):

            start = {k: v for k, v in init_start.items()}
            start['feat'] = self.wm.rssm.get_feat(start)  

            
            start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)


            if self.gc_input == "state" or self.config.if_actor_gs:
                goal_input = tf.cast(goal, self.dtype)  
            
            elif self.gc_input == "embed":
                goal_obs = start.copy()
                goal_obs[self.state_key] = goal
                goal_input = self.wm.encoder(goal_obs)  

            actor_inp = tf.concat([start['feat'], goal_input], -1)  

            start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
            seq = {k: [v] for k, v in start.items()}  

            
            for _ in range(self.horizon):
                actor_inp = tf.concat([seq['feat'][-1], goal_input], -1)
                action = self.actor(actor_inp).sample()
                state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
                feat = self.wm.rssm.get_feat(state)
                for key, value in {**state, 'action': action, 'feat': feat}.items():
                    seq[key].append(value)

            seq = {k: tf.stack(v, 0) for k, v in seq.items()}  
            # rewards should be (batch,1)
            rewards = self.reward_fn(seq)  

            # for k,v in seq.items():
            
            

            returns = tf.reduce_sum(rewards, 0)  

            
            # rewards = tf.ones([goal.shape[0],])
            return returns, seq

        # CEM loop
        # rewards = []
        # act_losses = []
        
        
        
        if self.init_distribution is None:
            # print("getting init distribtion from obs")
            means, stds = self.get_distribution_from_obs(obs)  
        else:
            # print("getting init distribtion from init candidates")
            means, stds = self.init_distribution

        # print(means, stds)
        opt_steps = 1 if self.evaluate_only else self.optimization_steps  

        
        for i in range(opt_steps):
            
            
            
            if i == 0 and (self.dataset or self.sample_env_goals):

                
                if self.dataset:
                    # print("getting init distribution from dataset")
                    random_batch = next(self.dataset)
                    random_batch = self.wm.preprocess(random_batch)

                    
                    samples = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
                    if self.obs2goal is not None:
                        samples = self.obs2goal(samples)

                
                elif self.sample_env_goals:
                    num_cem_samples = int(self.batch * self.env_goals_percentage)
                    num_env_samples = self.batch - num_cem_samples
                    cem_samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[num_cem_samples])
                    env_samples = self.sample_env_goals_fn(num_env_samples)
                    samples = tf.concat([cem_samples, env_samples], 0)

                # 
                # initialize means states.
                means, vars = tf.nn.moments(samples, 0)
                # stds = tf.sqrt(vars + 1e-6)
                # stds = tf.concat([[0.5, 0.5], stds[2:]], axis=0)
                # assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
                samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
                # print(i, samples)
                samples = tf.clip_by_value(samples, self.min_goal, self.max_goal)

            
            else:
                samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])  
                samples = tf.clip_by_value(samples, self.min_goal, self.max_goal)

            
            if self.repeat_samples > 1:
                repeat_samples = tf.repeat(samples, self.repeat_samples, 0)
                repeat_fitness, seq = eval_fitness(repeat_samples)
                fitness = tf.reduce_mean(tf.stack(tf.split(repeat_fitness, self.repeat_samples)), 0)
            else:
                fitness, seq = eval_fitness(samples)  

            # Refit distribution to elite samples
            
            if self.planner == 'shooting_mppi':
                # MPPI
                weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)
                means = tf.reduce_sum(weights * samples, axis=0)
                stds = tf.sqrt(tf.reduce_sum(weights * tf.square(samples - means), axis=0))
                # rewards.append(tf.reduce_sum(fitness * weights[:, 0]).numpy())

            
            elif self.planner == 'shooting_cem':
                # CEM
                elite_score, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
                elite_samples = tf.gather(samples, elite_inds)
                # print(elite_samples)
                means, vars = tf.nn.moments(elite_samples, 0)
                stds = tf.sqrt(vars + 1e-6)
                # rewards.append(tf.reduce_mean(tf.gather(fitness, elite_inds)).numpy())

        if self.planner == 'shooting_cem':
            self.vis_fn(elite_inds, elite_samples, seq, self.wm)
            self.elite_inds = elite_inds
            self.elite_samples = elite_samples
            self.final_seq = seq

        elif self.planner == 'shooting_mppi':
            # print("mppi mean", means)
            # print("mppi std", stds)
            # TODO: figure out what elite inds means for shooting mppi.
            # self.vis_fn(elite_inds, elite_samples, seq, self.wm)
            self.elite_inds = None
            self.elite_samples = None
            self.final_seq = seq

        # TODO: potentially store these as initialization for the next update.
        self.means = means
        self.stds = stds

        if self.evaluate_only:
            self.elite_samples = elite_samples
            self.elite_score = elite_score

        return self.sample_goal()  


    
    def sample_goal(self, batch=1):

        
        
        if self.evaluate_only:
            # samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
            # weights = tf.nn.softmax(self.elite_score)
            weights = self.elite_score / self.elite_score.sum()
            idxs = tf.squeeze(tf.random.categorical(tf.math.log([weights]), batch), 0)
            samples = tf.gather(self.elite_samples, idxs)

        
        else:
            samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])

        
        return samples


    
    def create_init_distribution(self, init_candidates):
        """Create the starting distribution for seeding the planner.
        """
        def _create_init_distribution(init_candidates):
            means = tf.reduce_mean(init_candidates, 0)
            stds = tf.math.reduce_std(init_candidates, 0)
            # if there's only 1 candidate, set std to default
            if init_candidates.shape[0] == 1:
                stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
            return means, stds
        
        self.init_distribution = _create_init_distribution(init_candidates)


    
    def get_distribution_from_obs(self, obs):
        ob = tf.squeeze(obs[self.state_key])
        if self.gc_input == "state" or self.config.if_actor_gs:
            ob = self.obs2goal(ob)
        means = tf.cast(tf.identity(ob), tf.float32)
        assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
        stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
        init_distribution = tf.identity(means), tf.identity(stds)
        return init_distribution


    
    def get_init_distribution(self):
        if self.init_distribution is None:
            means = tf.zeros(self.goal_dim, dtype=tf.float32)
            stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
            self.init_distribution = tf.identity(means), tf.identity(stds)

        return self.init_distribution


# CE2
class Cluster_goal_Planner:

    def __init__(self, 
                agnt, 
                config, 
                env,
                obs2goal_fn=None,):

        # agnt things
        self.agnt = agnt
        self.config = config
        self.wm = agnt.wm
        self.obs2goal_fn = obs2goal_fn
        self.cluster = agnt.wm.cluster
        self.dtype = agnt.wm.dtype
        if self.config.if_actor_gs:
            self.actor = agnt._task_behavior.actor_gs
        else:
            self.actor = agnt._task_behavior.actor
        

        if self.config.expl_behavior == 'Plan2Explore':
            self.reward_fn = agnt._expl_behavior.planner_intr_reward

        # p_cfg
        p_cfg = config.planner
        self.planner = p_cfg.planner_type
        self.horizon = p_cfg.horizon
        self.batch = p_cfg.batch
        self.cem_elite_ratio = p_cfg.cem_elite_ratio
        self.optimization_steps = p_cfg.optimization_steps
        self.std_scale = p_cfg.std_scale
        self.mppi_gamma = p_cfg.mppi_gamma
        self.evaluate_only = p_cfg.evaluate_only  
        self.repeat_samples = p_cfg.repeat_samples  
        self.env_goals_percentage = p_cfg.init_env_goal_percent
        
        # env
        self.goal_dim = np.prod(env.obs_space[config.goal_key].shape)
        self.act_space = env.act_space
        if isinstance(self.act_space, dict):
            self.act_space = self.act_space['action']

        # config
        self.gc_input = config.gc_input
        self.state_key = config.state_key

        
        state = None
        if state is None:
            self.initial_latent = self.wm.rssm.initial(1)
            self.initial_action = tf.zeros((1,) + self.act_space.shape)

        self.decoder = self.wm.heads['decoder']

        self.if_eval_fitness = False
        self.candidate_num = 1000
        self.sample_num = 100


    
    def search_goal(self, obs, state=None, **kwargs):

        
        if state is None:
            latent = self.wm.rssm.initial(1)
            action = tf.zeros((1, 1,) + self.act_space.shape)
            state = latent, action
            # print("make new state")
        else:
            latent, action = state
            action = tf.expand_dims(action, 0)
            # action should be (1, 1, D)
            # print("using exisitng state")

        
        # create start state.
        embed = self.wm.encoder(obs)
        # posterior is q(s' | s,a,e)
        post, prior = self.wm.rssm.observe(embed, action, obs['is_first'], latent)  # post: {''stoch': z_t+1, 'deter': h_t+1}
        # for k,v in post.items():
        
        init_start = {k: v[:, -1] for k, v in post.items()}  # (1, 1, d) --> (1, d)
        
        # for k,v in latent.items():
        

        
        @tf.function
        def eval_fitness(goal):

            start = {k: v for k, v in init_start.items()}
            start['feat'] = self.wm.rssm.get_feat(start)  

            
            start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)

            # print("dyl-1", start['feat'])
            # print("dyl-2", goal)

            actor_inp = tf.concat([start['feat'], goal], -1)  

            start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
            seq = {k: [v] for k, v in start.items()}  

            
            for _ in range(self.horizon):
                actor_inp = tf.concat([seq['feat'][-1], goal], -1)

                action = self.actor(actor_inp).sample()
                state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
                feat = self.wm.rssm.get_feat(state)
                for key, value in {**state, 'action': action, 'feat': feat}.items():
                    seq[key].append(value)

            seq = {k: tf.stack(v, 0) for k, v in seq.items()}  
            # rewards should be (batch,1)
            rewards = self.reward_fn(seq)  

            # for k,v in seq.items():
            
            

            returns = tf.reduce_sum(rewards, 0)  

            
            # rewards = tf.ones([goal.shape[0],])
            return returns, seq

        samples = self.cluster.sample(self.candidate_num, self.sample_num)


        # print(samples.shape)
        samples = tf.convert_to_tensor(samples.numpy(), dtype=self.dtype)


        if self.config.gc_input == "state" or self.config.if_actor_gs:

            initial_latent = tf.nest.map_structure(lambda x: tf.repeat(x, samples.shape[0], 0), self.initial_latent)
            initial_action = tf.nest.map_structure(lambda x: tf.repeat(x, samples.shape[0], 0), self.initial_action)


            
            latent, _ = self.wm.rssm.obs_step(initial_latent, initial_action, samples, True, True)

            feat = self.wm.rssm.get_feat(latent)

            # print("feat", feat)

            samples_decoded_dist = self.decoder(feat)

            samples_decoded = samples_decoded_dist[self.wm.state_key].mean()

            samples_decoded = self.obs2goal_fn(samples_decoded)

            samples_decoded = tf.cast(samples_decoded, dtype=self.dtype)

            if self.config.expl_behavior == 'Plan2Explore' and self.if_eval_fitness:

                fitness, seq = eval_fitness(samples_decoded)
                weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)

                max_indices = int(tf.argmax(weights).numpy())
                # print(max_indices)
                # print(weights[max_indices])
                explore_goal_decoded = samples_decoded[max_indices]

            else:

                explore_goal_decoded = random.choice(samples_decoded)


        else:

            
            if self.config.expl_behavior == 'Plan2Explore' and self.if_eval_fitness:

                fitness, seq = eval_fitness(samples)
                weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)

                max_indices = int(tf.argmax(weights).numpy())
                # print(max_indices)
                # print(weights[max_indices])
                explore_goal = samples[max_indices]
                # print(weights)
                # print(weights.shape)
            else:    
                explore_goal = random.choice(samples)


            
            explore_goal = tf.convert_to_tensor(explore_goal.numpy(), dtype=self.wm.dtype)
            explore_goal = explore_goal[None]

            # print("goal-picker")
            # print(explore_goal)
            # print(self.initial_latent)
            # print(self.initial_action)
            latent, _ = self.wm.rssm.obs_step(self.initial_latent, self.initial_action, explore_goal, True, True)

            feat = self.wm.rssm.get_feat(latent)

            # print("feat", feat)

            explore_goal_decoded_dist = self.decoder(feat)

            explore_goal_decoded = explore_goal_decoded_dist[self.wm.state_key].mean()


        return explore_goal_decoded



class Demo_goal_Planner:

    def __init__(self, 
                agnt, 
                config, 
                env,
                obs2goal_fn=None,):

        # agnt things
        self.agnt = agnt
        self.actor = agnt._task_behavior.actor
        if config.expl_behavior == 'Plan2Explore':
            self.reward_fn = agnt._expl_behavior.planner_intr_reward

        self.act_space = agnt.act_space
        self.config = config
        self.gc_input = config.gc_input
        p_cfg = config.planner
        self.horizon = p_cfg.horizon
        self.mppi_gamma = p_cfg.mppi_gamma
        self.env = env
        self.wm = agnt.wm
        self.obs2goal_fn = obs2goal_fn
        self.dtype = agnt.wm.dtype
        self.if_use_demo = config.if_use_demo
        assert self.if_use_demo
        self.demo_path = config.demo_path

        self.learning_rate = 0

        self.demo_search_strategy = self.config.demo_search_strategy

        self.search_repo = {}

        self.demo_wm_error_metric = 0

        self.if_eval_peg = True
        self.demo_peg_record = {}


    def search_goal(self, env, **kwargs):

        # target_demo_episode = env.goal_idx

        # if target_demo_episode not in self.search_repo:
        #     self.search_repo[target_demo_episode] = [True, None]  # [if_search_goal, goal]
        
        # if self.search_repo[target_demo_episode][0] or self.search_repo[target_demo_episode][1] is None:
        #     self.search_repo[target_demo_episode][0] = False
        # else:
        #     return self.search_repo[target_demo_episode][1]


        
        if self.demo_search_strategy == 0:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            index = int(length * self.learning_rate)
            
            
            
            start = max(0, index - 1)
            end = min(length, index + 2)
            
            goal = random.choice(demo_observations[start:end])
        
        
        elif self.demo_search_strategy == 1:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[1:]

            
            avg_value = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore_first > avg_value, tf.int32))

            
            index = index + 1

            
            goal = truth[0][index.numpy()]

        
        elif self.demo_search_strategy == 2:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)

            # self.learning_rate = 0.99
            
            
            start_index = max(int(length * self.learning_rate), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat[2:])

            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            avg_value = tf.reduce_mean(demo_prediction_error_list_flat_ignore)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > avg_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]

        
        elif self.demo_search_strategy == 3:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            start_index = max(int(length * self.learning_rate), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            avg_value = tf.reduce_mean(demo_prediction_error_list_flat_ignore)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > avg_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]

        # Multi-step imagine
        elif self.demo_search_strategy == 4:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            start_index = max(int(length * self.learning_rate), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm_2(demo_trajectory, self.wm)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            avg_value = tf.reduce_mean(demo_prediction_error_list_flat_ignore)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > avg_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]
            
        # ten-step imagine for each step
        elif self.demo_search_strategy == 5:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            start_index = max(int(length * self.learning_rate), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm_3(demo_trajectory, self.wm)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            avg_value = tf.reduce_mean(demo_prediction_error_list_flat_ignore)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > avg_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]

        
        elif self.demo_search_strategy == 6:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            start_index = max(int(length * self.learning_rate), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm)  

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            median_value = tfp.stats.percentile(demo_prediction_error_list_flat_ignore, 50.0)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > median_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]

        
        elif self.demo_search_strategy == 7:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            start_index = max(int(length * self.learning_rate), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm)  

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            quarter_percentile_value = tfp.stats.percentile(demo_prediction_error_list_flat_ignore, 25.0)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > quarter_percentile_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]

        
        elif self.demo_search_strategy == 8:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            error_rank = max(int(length * max(self.learning_rate, 0.4)), 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm)  

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[1:]

            values, indices = tf.math.top_k(demo_prediction_error_list_flat_ignore_first, k=(length-error_rank+1))  

            
            index = indices[-1]

            # print(indices)

            index = index + 1
            # print("index", index)

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            goal = truth[0][index.numpy()]

        
        elif self.demo_search_strategy == 9:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)
            
            
            start_index = max(int(length * min(self.learning_rate+0.2, 1)) - 1, 1)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[start_index:]

            
            avg_value = tf.reduce_mean(demo_prediction_error_list_flat_ignore)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore > avg_value, tf.int32))

            
            index = index + start_index

            
            goal = truth[0][index.numpy()]


        
        elif self.demo_search_strategy == 10:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(
                demo_trajectory, self.wm, use_post_states=True
            )  # Ensure deep copy inside the function to avoid modifying the original values

            # Reshape demo_prediction_error_list to 1D
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            # Ignore first value because the first imagined state is generated from initial zero variables and has no reference value
            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[2:]


            normalized_errors = demo_prediction_error_list_flat_ignore / tf.reduce_sum(demo_prediction_error_list_flat_ignore)
    
            normalized_probs = normalized_errors.numpy()
            chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            # Add start_index to match the original tensor indexing
            index = chosen_idx + 2

            # Get the corresponding truth value
            goal = truth[0][index]


        
        elif self.demo_search_strategy == 11:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_observations = demo_trajectory[self.config.state_key]
            length = len(demo_observations)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(
                demo_trajectory, self.wm, use_post_states=True
            )  # Ensure deep copy inside the function to avoid modifying the original values

            # Reshape demo_prediction_error_list to 1D
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            # Ignore first value because the first imagined state is generated from initial zero variables and has no reference value
            demo_prediction_error_list_flat_ignore = demo_prediction_error_list_flat[2:]

            # Square the values
            squared_errors = tf.square(demo_prediction_error_list_flat_ignore)

            normalized_squared_errors = squared_errors / tf.reduce_sum(squared_errors)
    
            normalized_probs = normalized_squared_errors.numpy()
            chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            # Add start_index to match the original tensor indexing
            index = chosen_idx + 2

            # Get the corresponding truth value
            goal = truth[0][index]
            

        
        elif self.demo_search_strategy == 12:

            error_threshold = 0.02

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore_first > error_threshold, tf.int32))

            
            index = index + 2

            min_index = max(0, index)
            max_index = min(len(truth[0]), index + 20)

            goal_index = random.choice(range(min_index, max_index))

            
            goal = truth[0][goal_index]


        
        elif self.demo_search_strategy == 13:

            error_threshold = 0.03

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore_first > error_threshold, tf.int32))

            
            index = index + 2

            goal_index = random.choice(range(index, len(truth[0]) -1 ))

            
            goal = truth[0][goal_index]


        
        elif self.demo_search_strategy == 14:

            error_threshold = 0.03

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore_first > error_threshold, tf.int32))

            
            index = index + 2

            demo_prediction_error_list_flat_range = demo_prediction_error_list_flat[index:]
            normalized_demo_prediction_error_list_flat_range = demo_prediction_error_list_flat_range / tf.reduce_sum(demo_prediction_error_list_flat_range)
    
            
            # chosen_idx = np.random.choice(len(normalized_demo_prediction_error_list_flat_range), p=normalized_demo_prediction_error_list_flat_range)
            normalized_probs = normalized_demo_prediction_error_list_flat_range.numpy()
            chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            goal_index = min(index + chosen_idx, len(truth[0]) - 1)

            
            goal = truth[0][goal_index]


        
        elif self.demo_search_strategy == 15:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            
            quarter_percentile_value = tfp.stats.percentile(demo_prediction_error_list_flat_ignore_first, 75.0)

            error_threshold = quarter_percentile_value

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            
            index = tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore_first > error_threshold, tf.int32))

            
            index = index + 2

            demo_prediction_error_list_flat_range = demo_prediction_error_list_flat[index:]
            normalized_demo_prediction_error_list_flat_range = demo_prediction_error_list_flat_range / tf.reduce_sum(demo_prediction_error_list_flat_range)
    
            
            # chosen_idx = np.random.choice(len(normalized_demo_prediction_error_list_flat_range), p=normalized_demo_prediction_error_list_flat_range)
            normalized_probs = normalized_demo_prediction_error_list_flat_range.numpy()
            chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            goal_index = min(index + chosen_idx, len(truth[0]) - 1)

            
            goal = truth[0][goal_index]


        
        elif self.demo_search_strategy == 16:


            if self.if_eval_peg:

                self.demo_peg_record = {}

                self.if_eval_peg = False

                all_demo_tra, tra_num = self.get_all_demo_trajectory_by_env(env)

                for i in range(tra_num):

                    state = None

                    demo_trajectory = all_demo_tra[i]

                    
                    if state is None:
                        latent = self.wm.rssm.initial(1)
                        action = tf.zeros((1, 1,) + self.act_space.shape)
                        state = latent, action
                        # print("make new state")
                    else:
                        latent, action = state
                        action = tf.expand_dims(action, 0)
                        # action should be (1, 1, D)
                        # print("using exisitng state")

                    
                    # create start state.
                    obs = {}
                    for key, value in demo_trajectory.items():
                        obs[key] = np.expand_dims(np.expand_dims(value[0], axis=0), axis=0)
                    embed = self.wm.encoder(obs)
                    # posterior is q(s' | s,a,e)
                    post, prior = self.wm.rssm.observe(embed, action, obs['is_first'], latent)  # post: {''stoch': z_t+1, 'deter': h_t+1}
                    # for k,v in post.items():
                    
                    init_start = {k: v[:, -1] for k, v in post.items()}  # (1, 1, d) --> (1, d)
                    
                    # for k,v in latent.items():
                    
                    
                    
                    @tf.function
                    def eval_fitness(goal):

                        start = {k: v for k, v in init_start.items()}
                        start['feat'] = self.wm.rssm.get_feat(start)  

                        
                        start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)


                        if self.gc_input == "state":
                            goal_input = tf.cast(goal, self.dtype)  
                        
                        elif self.gc_input == "embed":
                            goal_obs = start.copy()
                            goal_obs[self.config.state_key] = goal
                            goal_input = self.wm.encoder(goal_obs)  

                        actor_inp = tf.concat([start['feat'], goal_input], -1)  

                        start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
                        seq = {k: [v] for k, v in start.items()}  

                        
                        for _ in range(self.horizon):
                            actor_inp = tf.concat([seq['feat'][-1], goal_input], -1)
                            action = self.actor(actor_inp).sample()
                            state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
                            feat = self.wm.rssm.get_feat(state)
                            for key, value in {**state, 'action': action, 'feat': feat}.items():
                                seq[key].append(value)

                        seq = {k: tf.stack(v, 0) for k, v in seq.items()}  
                        # rewards should be (batch,1)
                        rewards = self.reward_fn(seq)  

                        # for k,v in seq.items():
                        
                        

                        returns = tf.reduce_sum(rewards, 0)  

                        
                        # rewards = tf.ones([goal.shape[0],])
                        return returns, seq


                    demo_obs = list(demo_trajectory[self.config.state_key])
                    samples = tf.convert_to_tensor(demo_obs, dtype=self.dtype)
                    fitness, seq = eval_fitness(samples)
                    weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)

                    self.demo_peg_record[i] = weights


            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)
            demo_obs = list(demo_trajectory[self.config.state_key])
            demo_peg_weights = self.demo_peg_record[env.goal_idx]
            max_indices = int(tf.argmax(demo_peg_weights).numpy())
            goal = demo_obs[max_indices]


        
        elif self.demo_search_strategy == 17:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            
            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(
                demo_trajectory, self.wm, use_post_states=False
            )

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            
            half_len = tf.cast(tf.shape(demo_prediction_error_list_flat_ignore_first)[0] // 4, tf.int32)
            first_half = demo_prediction_error_list_flat_ignore_first[:half_len]
            threshold = tf.reduce_max(first_half)

            
            global_max = tf.reduce_max(demo_prediction_error_list_flat_ignore_first)

            
            index = tf.cond(
                tf.equal(threshold, global_max),
                lambda: tf.argmax(demo_prediction_error_list_flat_ignore_first),  
                lambda: tf.argmax(tf.cast(demo_prediction_error_list_flat_ignore_first > threshold, tf.int32))  
            )

            if index == 0:

                index = 0

            
            index = index + 2

            
            min_index = max(0, index)
            max_index = min(len(truth[0]), index + 5)

            
            if random.random() < 0.2:
                
                goal_index = len(truth[0]) - 1
            else:
                
                goal_index = random.choice(range(min_index, max_index))

            
            goal = truth[0][goal_index]

        
        elif self.demo_search_strategy == 18:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_obs = demo_trajectory[self.config.state_key]

            
            eval_value = self.Eval_demotra_by_P2E(demo_trajectory, self.wm)

            
            eval_value_flat = tf.reshape(eval_value, [-1])

            
            eval_value_flat_ignore_first = eval_value_flat[2:]

            self.demo_wm_error_metric = tf.reduce_mean(eval_value_flat_ignore_first)

            # Square the values
            squared_value = tf.exp(eval_value_flat_ignore_first)

            normalized_squared_value = squared_value / tf.reduce_sum(squared_value)
    
            normalized_probs = normalized_squared_value.numpy()
            chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            # Add start_index to match the original tensor indexing
            index = chosen_idx + 2


            
            if random.random() < 0.2:
                
                goal_index = len(demo_obs) - 1
            else:
                
                goal_index = index

            
            goal = demo_obs[goal_index]


        
        elif self.demo_search_strategy == 19:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            truth, imagined_decoded, demo_prediction_error_list = self.Predict_demotra_by_wm(demo_trajectory, self.wm, use_post_states=True)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            normalized_demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat_ignore_first / tf.reduce_sum(demo_prediction_error_list_flat_ignore_first)
    
            
            # chosen_idx = np.random.choice(len(normalized_demo_prediction_error_list_flat_range), p=normalized_demo_prediction_error_list_flat_range)
            normalized_probs = normalized_demo_prediction_error_list_flat_ignore_first.numpy()
            chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            # Add start_index to match the original tensor indexing
            index = chosen_idx + 2

            
            if random.random() < 0.2:
                
                goal_index = len(truth[0]) - 1
            else:
                
                goal_index = index

            
            goal = truth[0][goal_index]


        
        elif self.demo_search_strategy == 20:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_obs = demo_trajectory[self.config.state_key]

            dd = self.Eval_demotra_by_dd(demo_trajectory, self.wm)  

            # print("demo_prediction_error_list:", demo_prediction_error_list)

            
            # demo_prediction_error_list_flat = tf.reshape(demo_prediction_error_list, [-1])

            
            # demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat[2:]

            # self.demo_wm_error_metric = tf.reduce_mean(demo_prediction_error_list_flat_ignore_first)

            # normalized_demo_prediction_error_list_flat_ignore_first = demo_prediction_error_list_flat_ignore_first / tf.reduce_sum(demo_prediction_error_list_flat_ignore_first)
    
            
            # # chosen_idx = np.random.choice(len(normalized_demo_prediction_error_list_flat_range), p=normalized_demo_prediction_error_list_flat_range)
            # normalized_probs = normalized_demo_prediction_error_list_flat_ignore_first.numpy()
            # chosen_idx = np.random.choice(len(normalized_probs), p=normalized_probs)

            chosen_idx = np.argmax(dd)
            # Add start_index to match the original tensor indexing
            index = chosen_idx + 2


            
            if random.random() < 0.2:
                
                goal_index = len(demo_obs) - 1
            else:
                
                goal_index = index

            
            goal = demo_obs[goal_index]


        
        elif self.demo_search_strategy == 21:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_obs = demo_trajectory['observation']

            demo_obs_count_list = self.env.demo_obs_count_dict[self.env.goal_idx]

            # demo_obs_count_mean = np.mean(demo_obs_count_list)
            filtered_list = [x for x in demo_obs_count_list if x != 0]

            
            if len(filtered_list) == 0:
                demo_obs_count_median = 0
            else:
                demo_obs_count_median = np.median(filtered_list)

            threshold = np.clip(demo_obs_count_median, self.config.count_lb, self.config.count_ub)  # lower bound and upper bound
            # threshold = np.int32(100)

            
            greater_indices = np.where(demo_obs_count_list >= threshold)[0]

            if len(greater_indices) > 0 and threshold > 0:
                index = greater_indices[-1] + 1
            else:
                index = 0

            
            min_index = max(0, index - max(len(demo_obs) // self.config.sample_range, 1))
            max_index = min(len(demo_obs), index + max(len(demo_obs) // self.config.sample_range, 1))

            
            final_goal_rate = self.config.final_goal_sample_rate
            if random.random() < final_goal_rate:
                
                goal_index = len(demo_obs) - 1
            else:
                
                # print("min_index:", min_index, "max_index:", max_index, "demo_obs length:", len(demo_obs))
                goal_index = random.choice(range(min_index, max_index))

            # goal_index = int(len(demo_obs) * 0.4)

            goal = demo_obs[goal_index]

            if self.config.if_image_obs:

                image_goal = demo_trajectory['images'][goal_index]

                return goal, image_goal

        
        elif self.demo_search_strategy == 22:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_obs = demo_trajectory[self.config.state_key]

            goal_index = random.choice(range(0, len(demo_obs) - 1))

            goal = demo_obs[goal_index]
        

        # final goal
        elif self.demo_search_strategy == 23:

            demo_trajectory = self.sample_demo_trajectory_by_env_seed(env)

            demo_obs = demo_trajectory[self.config.state_key]

            # goal_index = random.choice(range(0, len(demo_obs) - 1))

            goal = demo_obs[-1]

        return goal


    @tf.function
    # one step imagine
    def Predict_demotra_by_wm(self, demo_trajectory, wm, use_post_states=False):
        
        demo_tra = {key: tf.identity(value) for key, value in demo_trajectory.items()}

        square_error_list = []
        decoder = wm.heads['decoder']  

        for key, value in demo_tra.items():
            demo_tra[key] = value[None, :]  

        truth = tf.cast(demo_tra[self.config.state_key], tf.float32)  
        embed = wm.encoder(demo_tra)

        
        post_states, prior_states = wm.rssm.observe(embed[:, :], demo_tra['action'][:, :], demo_tra['is_first'][:, :])  # one step imagine

        if use_post_states:
            imagined_decoded = tf.cast(decoder(wm.rssm.get_feat(post_states))[self.config.state_key].mode()[:], tf.float32)
        else:
            imagined_decoded = tf.cast(decoder(wm.rssm.get_feat(prior_states))[self.config.state_key].mode()[:], tf.float32)  

        # tf.print("Imagined Decoded:", imagined_decoded, summarize=-1)
        
        square_error_list.append(tf.reduce_sum(tf.square(truth - imagined_decoded), axis=2))

        return truth, imagined_decoded, square_error_list
    
    @ tf.function
    # totally imagine from the start
    def Predict_demotra_by_wm_2(self, demo_trajectory, wm):
        
        demo_tra = {key: tf.identity(value) for key, value in demo_trajectory.items()}

        square_error_list = []
        decoder = wm.heads['decoder']  

        for key, value in demo_tra.items():
            demo_tra[key] = value[None, :]  

        truth = tf.cast(demo_tra[self.config.state_key], tf.float32)  
        embed = wm.encoder(demo_tra)

        
        post_states, prior_states = wm.rssm.observe(embed[: , :2], demo_tra['action'][:, :2], demo_tra['is_first'][:, :2])  
        init = {k: v[:, -1] for k, v in post_states.items()}  
        prior = wm.rssm.imagine(demo_tra['action'][: , 2:], init)  
        totally_imagined_decoded = tf.cast(decoder(wm.rssm.get_feat(prior))[self.config.state_key].mode()[:], tf.float32)

        square_error_list.append(tf.reduce_sum(tf.square(truth[:, 2:, :] - totally_imagined_decoded), axis=2))

        return truth[:, 2:, :], totally_imagined_decoded, square_error_list

    @ tf.function
    # ten-step imagine for each step
    def Predict_demotra_by_wm_3(self, demo_trajectory, wm):
        # Create a new dictionary to store the copied demo_trajectory data
        demo_tra = {key: tf.identity(value) for key, value in demo_trajectory.items()}

        error_list = []
        decoder = wm.heads['decoder']  # Decoder

        for key, value in demo_tra.items():
            demo_tra[key] = value[None, :]  # Expand tensor along the batch dimension

        truth = tf.cast(demo_tra[self.config.state_key], tf.float32)  # Convert truth to float32
        embed = wm.encoder(demo_tra)

        # Loop over each step and imagine the next 10 steps (or fewer if near the end)
        total_steps = truth.shape[1]
        for t in range(2, total_steps):
            remaining_steps = min(10, total_steps - t)  # Adjust if fewer than 10 steps are left

            # Observe the current step to update init
            post_states, _ = wm.rssm.observe(embed[:, t:t+1], demo_tra['action'][:, t:t+1], demo_tra['is_first'][:, t:t+1])
            init = {k: v[:, -1] for k, v in post_states.items()}  # Update init for the current step

            # Imagine the next few steps starting from the updated init
            prior = wm.rssm.imagine(demo_tra['action'][:, t:t+remaining_steps], init)  # Imagine the next steps
            imagined_decoded = tf.cast(decoder(wm.rssm.get_feat(prior))[self.config.state_key].mode()[:], tf.float32)

            # Calculate the squared error for the imagined steps
            error = tf.reduce_mean(tf.square(truth[:, t:t+remaining_steps, :] - imagined_decoded), axis=[1, 2])
            error_list.append(error)

        # Return truth and the list of errors for each step
        return truth[:, 2:, :], None, [error_list]
        
    @ tf.function
    def Eval_demotra_by_P2E(self, demo_trajectory, wm):

        
        demo_tra = {key: tf.identity(value) for key, value in demo_trajectory.items()}


        for key, value in demo_tra.items():
            demo_tra[key] = value[None, :]  

        embed = wm.encoder(demo_tra)

        
        post_states, prior_states = wm.rssm.observe(embed[:, :], demo_tra['action'][:, :], demo_tra['is_first'][:, :])  # one step imagine

        feat = wm.rssm.get_feat(post_states)

        demo_tra['feat'] = feat

        
        eval_value = self.agnt._expl_behavior._intr_reward(demo_tra)

        return eval_value


    def Eval_demotra_by_dd(self, demo_trajectory, wm):
        
        demo_tra = {key: tf.identity(value) for key, value in demo_trajectory.items()}

        for key, value in demo_tra.items():
            demo_tra[key] = value[None, :]  

        
        embed = wm.encoder(demo_tra)  

        post_states, prior_states = wm.rssm.observe(embed[:, :], demo_tra['action'][:, :], demo_tra['is_first'][:, :])  

        feat = wm.rssm.get_feat(prior_states)

        imgined_embed = wm.heads['embed'](feat).mode()

        inp_embed = tf.cast(tf.squeeze(embed, axis=0), tf.float32)  
        goal_embed = tf.cast(tf.squeeze(imgined_embed, axis=0), tf.float32)  

        
        dd_pred = self.agnt._task_behavior.dynamical_distance(tf.concat([inp_embed, goal_embed], axis=-1))  # shape: (num_segments, 2*D)


        # totally_imagined_decoded = tf.cast(decoder(wm.rssm.get_feat(prior))[self.config.state_key].mode()[:], tf.float32)

        # square_error_list.append(tf.reduce_sum(tf.square(truth[:, 2:, :] - totally_imagined_decoded), axis=2))


        return dd_pred
    

    def sample_demo_trajectory_by_env_seed(self, env):

        target_demo_episode = env.goal_idx  

        env_demo_tra = env.all_demo_trajectories[target_demo_episode]

        if self.config.if_image_obs:

            target_key  = ['observation', 'action', 'reward', 'is_last', 'is_first', 'images']

        else:
            target_key  = ['observation', 'action', 'reward', 'is_last', 'is_first']

        demo_tra = {key: env_demo_tra[key] for key in target_key}

        return demo_tra


    def get_all_demo_trajectory_by_env(self, env):

        all_demo_trajectories = {}

        for i in range(len(env.all_demo_trajectories)):

            env_demo_tra = env.all_demo_trajectories[i]

            target_key  = ['observation', 'action', 'reward', 'is_last', 'is_first']

            demo_tra = {key: env_demo_tra[key] for key in target_key}

            all_demo_trajectories[i] = demo_tra

        return all_demo_trajectories, len(env.all_demo_trajectories)


def softmax(X, theta=1.0, axis=None):
    """
        Compute the softmax of each element along an axis of X.

        Parameters
        ----------
        X: ND-Array. Probably should be floats.
        theta (optional): float parameter, used as a multiplier
                prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the
                first non-singleton axis.

        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.max(y, axis=axis, keepdims=True)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.sum(y, axis=axis, keepdims=True)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


















