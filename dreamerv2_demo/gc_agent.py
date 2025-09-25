

import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import numpy as np

import common
import dreamerv2_demo.explorer as explorer
import dreamerv2_demo.nor_agent as nor_agent
import sys
import os

from torch.optim import Adam
import torch
import random
import pathlib
from dreamerv2_demo.Goal_Predictor import get_demo_trajectories



class GCAgent(common.Module):

    def __init__(self, config, obs_space, act_space, step, obs2goal, sample_env_goals):
        self.config = config
        # TODO: assumes we're doing state based envs.
        self.state_key = config.state_key
        self.goal_key = config.goal_key
        self.obs_space = obs_space
        
        goal_dim = np.prod(self.obs_space[self.goal_key].shape)

        

        # self.obs_space.pop(self.goal_key)

        self.act_space = act_space['action']
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)
        self.wm = GCWorldModel(config, obs_space, self.tfstep, obs2goal, sample_env_goals)
        self._task_behavior = GCActorCritic(config, self.act_space, self.tfstep, obs2goal, goal_dim)
        if config.expl_behavior == 'greedy':
            self._expl_behavior = self._task_behavior
        elif config.expl_behavior == 'gc-explorer':
            self._expl_behavior = GCActorCritic_Explorer(config, self.act_space, self.wm, self.tfstep, obs2goal, goal_dim)
        else:
            self._expl_behavior = getattr(explorer, config.expl_behavior)(
                    self.config, self.act_space, self.wm, self.tfstep,
                    lambda seq: self.wm.heads['reward'](seq['feat']).mode())
            
        if config.if_goal_optimizer:
            self.goal_optimizer = Goal_optimizer(config, self.tfstep, goal_dim)

    
    @tf.function
    def expl_policy(self, obs, state=None, mode='train'):
        if self.config.expl_behavior == 'greedy':
            return self.policy(obs, state, mode)

        # run the plan2expl policy (not goal cond.)
        obs = tf.nest.map_structure(tf.tensor, obs)
        tf.py_function(lambda: self.tfstep.assign(
                int(self.step), read_value=False), [], [])
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
            state = latent, action
        latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        sample = (mode == 'train') or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample)
        feat = self.wm.rssm.get_feat(latent)

        if self.config.expl_behavior == 'gc-explorer':
            obs = self.wm.preprocess(obs)
            goal = self.wm.get_goal(obs, training=False, goal_from_env = False if mode=='train' else True) # just use current goal from obs
            actor_inp = tf.concat([feat, goal], -1)
            actor = self._expl_behavior.actor(actor_inp)

        elif self.config.expl_behavior == 'Demo_Explorer' or self.config.expl_behavior == 'Demo_BC_Explorer' or self.config.expl_behavior == 'Gail_Explorer':
            obs = self.wm.preprocess(obs)
            # goal = self.wm.get_goal(obs, training=False, goal_from_env = False if mode=='train' else True) # just use current goal from obs
            actor = self._expl_behavior.actor(obs['observation'])
        
        else:
            
            actor = self._expl_behavior.actor(feat)

            
        action = actor.sample()
        # action = actor.mode()
        noise = self.config.expl_noise
        action = common.action_noise(action, noise, self.act_space)
        outputs = {'action': action}
        state = (latent, action)
        return outputs, state

    
    
    @tf.function
    def policy(self, obs, state=None, mode='train'):
        obs = tf.nest.map_structure(tf.tensor, obs)
        obs = self.wm.preprocess(obs)
        assert mode in {'train', 'eval'}
         # use given goal

        
        
        goal = self.wm.get_goal(obs, training=False, goal_from_env = False if mode=='train' else True) # just use current goal from obs

        

        tf.py_function(lambda: self.tfstep.assign(
                int(self.step), read_value=False), [], [])
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
            state = latent, action
        latent, action = state
        embed = self.wm.encoder(obs)
        sample = (mode == 'train') or not self.config.eval_state_mean
        # sample = False

        # ACH Add
        # if mode == 'eval':
        #     sample = False

        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample)
        feat = self.wm.rssm.get_feat(latent)  # latent:{''stoch': z, 'deter': h}; feat: (z + h)
        actor_inp = tf.concat([feat, goal], -1)
        if mode == 'eval':
            actor = self._task_behavior.actor(actor_inp)
            action = actor.mode()
            noise = self.config.eval_noise  
        elif mode == 'explore':
            actor = self._expl_behavior.actor(actor_inp)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == 'train':
            actor = self._task_behavior.actor(actor_inp)
            action = actor.sample()
            noise = self.config.expl_noise
        if self.config.epsilon_expl_noise > 0 and mode != 'eval':
            action = common.epsilon_action_noise(action, self.config.epsilon_expl_noise, self.act_space)
        else:
            action = common.action_noise(action, noise, self.act_space)
        outputs = {'action': action}
        state = (latent, action)
        return outputs, state

    @tf.function
    def policy_gs(self, obs, state=None, mode='train'):

        obs = {key: value for key, value in obs.items() if not isinstance(value, dict)}
        obs = tf.nest.map_structure(tf.tensor, obs)
        obs = self.wm.preprocess(obs)
        assert mode in {'train', 'eval'}
         # use given goal

        
        goal = tf.cast(obs[self.goal_key], self.wm.dtype)

        tf.py_function(lambda: self.tfstep.assign(
                int(self.step), read_value=False), [], [])
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
            state = latent, action
        latent, action = state
        embed = self.wm.encoder(obs)
        sample = (mode == 'train') or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample)
        feat = self.wm.rssm.get_feat(latent)  # latent:{''stoch': z, 'deter': h}; feat: (z + h)
        actor_inp = tf.concat([feat, goal], -1)


        if mode == 'eval':
            actor = self._task_behavior.actor_gs(actor_inp)
            action = actor.mode()
            noise = self.config.eval_noise  

        elif mode == 'train':
            actor = self._task_behavior.actor_gs(actor_inp)
            action = actor.sample()
            noise = self.config.expl_noise

        if self.config.epsilon_expl_noise > 0 and mode != 'eval':
            action = common.epsilon_action_noise(action, self.config.epsilon_expl_noise, self.act_space)
        else:
            action = common.action_noise(action, noise, self.act_space)

        outputs = {'action': action}
        state = (latent, action)
        return outputs, state


    
    @tf.function
    def train_gcp(self, data, state=None):
        metrics = {}
        pdata = self.wm.preprocess(data)
        embed = self.wm.encoder(pdata)

        start, _ = self.wm.rssm.observe(embed, pdata['action'], pdata['is_first'], state)

        state = {k: v[:, -1] for k, v in start.items()}

        metrics.update(self._task_behavior.train(self.wm, start, data['is_terminal'], obs=data))

        return state, metrics

    @tf.function
    def train_wm(self, data, state=None):

        metrics = {}
        if self.config.if_opt_embed_by_dd:
            state, outputs, mets = self.wm.train(data, self._task_behavior.dynamical_distance, state)
        else:
            state, outputs, mets = self.wm.train(data, None, state)
        metrics.update(mets)

        return state, metrics
    

    
    @tf.function
    def train(self, data, state=None, train_cluster=False):

        
        metrics = {}
        if self.config.if_opt_embed_by_dd:
            state, outputs, mets = self.wm.train(data, self._task_behavior.dynamical_distance, state, train_cluster=train_cluster)
        else:
            state, outputs, mets = self.wm.train(data, None, state, train_cluster=train_cluster)
        metrics.update(mets)
        start = outputs['post']

        metrics.update(self._task_behavior.train(self.wm, start, data['is_terminal'], obs=data))

        if self.config.expl_behavior == 'gc-explorer':
            metrics.update(self._expl_behavior.train(self.wm, start, outputs, data['is_terminal'], data))

        else:
            if self.config.expl_behavior != 'greedy':
                mets = self._expl_behavior.train(start, outputs, data)[-1]
                metrics.update({'expl_' + key: value for key, value in mets.items()})

        return state, metrics

    
    # @tf.function
    def report(self, data, env, video_from_state_fn=None):
        report = {}
        data = self.wm.preprocess(data)

        
        if video_from_state_fn is not None:
            recon, openl, truth = self.wm.state_pred(data)
            report[f'openl_{self.state_key}'] = video_from_state_fn(recon, openl, truth, env)
        
        else: # image based env
            for key in self.wm.heads['decoder'].cnn_keys:
                name = key.replace('/', '_')

                
                report[f'openl_{name}'] = self.wm.video_pred(data, key)
    
        return report

    @tf.function
    def temporal_dist(self, obs):
        dist = self._task_behavior.subgoal_dist(self.wm, obs)
        if self.config.gc_reward == 'dynamical_distance' and self.config.dd_norm_reg_label:
                dist *= self._task_behavior.dd_seq_len
        return dist


    def agent_save(self, logdir):

        self.save(logdir / 'variables.pkl')

        if self.config.if_self_cluster:
            torch.save(self.wm.cluster.state_dict(), logdir / 'cluster.pth')


    def agent_load(self, logdir):

        self.load(logdir / 'variables.pkl')

        if self.config.if_self_cluster:
            self.wm.cluster.load_state_dict(torch.load(logdir / 'cluster.pth'))



class GCWorldModel(nor_agent.WorldModel):

    def __init__(self, config, obs_space, tfstep, obs2goal, sample_env_goals):
        self.shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.state_key = config.state_key
        self.goal_key = config.goal_key
        self.tfstep = tfstep
        self.obs2goal = obs2goal
        self.sample_env_goals = sample_env_goals
        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.encoder = common.Encoder(self.shapes, self.state_key, **config.encoder)
        self.embed_size = self.encoder.embed_size
        self.heads = {}
        self.heads['decoder'] = common.Decoder(self.shapes, **config.decoder)
        if config.pred_reward:
            self.heads['reward'] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads['discount'] = common.MLP([], **config.discount_head)
        if config.pred_embed:
            self.heads['embed'] = common.MLP([self.embed_size], **config.embed_head)
        
        # for name in config.grad_heads:
        #     assert name in self.heads, name
        self.model_opt = common.Optimizer('model', **config.model_opt)
        self.dtype = prec.global_policy().compute_dtype


        
        self.dd_cur_idxs, self.dd_goal_idxs = get_future_goal_idxs(seq_len = self.config.dataset.length, bs = self.config.dataset.batch)
        if self.config.if_self_dd_net:
            dd_out_dim = 1  
            self.dd_loss_fn = tf.keras.losses.MSE  # loss function
            self.dd_out_dim = dd_out_dim  
            
            self.L3P_dynamical_distance = common.L3P_GC_Distance(out_dim = dd_out_dim, input_type= self.config.dd_inp, units=400, normalize_input = self.config.dd_norm_inp)
            # print(self.config.dataset.batch)
            # print(self.config.dataset.length)
            # print(self.config.imag_horizon)
            
            
            
            
            
            self._L3P_dd_opt = common.Optimizer('dyn_dist', **config.dd_opt)

        # L3P Cluster
        if self.config.if_self_cluster:

            self.cluster = common.Cluster(config = config, embed_size = self.embed_size )
            self.c_optim = Adam(self.cluster.parameters(), lr=config.cluster['lr_cluster'])

        # try:
            
        #     self.all_demo_trajectories, self.seed_list = get_demo_trajectories(config.task, config.demo_path, if_eval=False, if_random_seeds=True, random_seeds_num=150)

        # except:

        #     print("Can't build demo_dataset in World Model")

        try:

            logdir = pathlib.Path(config.logdir).expanduser()
            demo_replay = common.Replay(logdir / 'demo_episodes', **config.replay)  # initialize replay buffer
            self.demo_dataset = iter(demo_replay.dataset(**config.dataset))

        except Exception as e:

            print("Can't build demo_dataset in GC_WM, because of ", e)

    @tf.function
    def train(self, data, dynamical_distance_net, state=None, train_cluster=False):

        data = self.preprocess(data)
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, dynamical_distance_net, state)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        metrics.update(self.model_opt(model_tape, model_loss, modules))

        if self.config.if_self_dd_net:
            
            with tf.GradientTape() as L3P_dd_net_tape:

                L3P_dd_net_loss, metrics_L3P_dd_net_loss = self.get_L3P_dd_net_loss(data[self.config.state_key])

            metrics.update(metrics_L3P_dd_net_loss)
            metrics.update(self._L3P_dd_opt(L3P_dd_net_tape, L3P_dd_net_loss, self.L3P_dynamical_distance))

        if self.config.if_self_cluster and train_cluster:

            embed = self.encoder(data)
            cluster_loss = tf.py_function(func=self.train_the_L3P_cluster, inp=[embed], Tout=tf.float32)
            metrics_cluster = {"cluster_loss": cluster_loss}
            metrics.update(metrics_cluster)


        return state, outputs, metrics
    

    def train_the_L3P_cluster(self, batch_embed):

        # with torch.no_grad():
        #     batch_embed = self.encoder(data)
        self.c_optim.zero_grad()
        loss_embed, metrics = self.cluster_loss(batch_embed)

        loss_embed.backward()
        self.c_optim.step()

        return loss_embed.detach().numpy()

    @staticmethod
    def _has_nan(x):
        return torch.any(torch.isnan(x)).cpu().numpy() == True
    

    def cluster_loss(self, embedding):

        
        
        numpy_embedding = embedding.numpy()

        

        torch_tensor_embedding = torch.Tensor(numpy_embedding)

        torch_tensor_embedding = torch_tensor_embedding.reshape(-1, self.config.encoder['mlp_layers'][-1])

        

        assert type(torch_tensor_embedding) == torch.Tensor

        # cluter_max_prob, cluster_prob = self.cluster.cluster_to_x(torch_tensor_embedding)
        
        
        
        
        

        posterior, elbo = self.cluster(torch_tensor_embedding, with_elbo=True)
        log_data = elbo['log_data']
        kl_from_prior = elbo['kl_from_prior']

        if self._has_nan(log_data) or self._has_nan(kl_from_prior):
            pass


        loss_elbo = - (log_data - self.config.cluster['lr_cluster'] * kl_from_prior).mean()
        std_mean = self.cluster.std_mean()
        loss_std = self.config.cluster['cluster_std_reg'] * std_mean

        loss_embed_total = loss_elbo + loss_std

        metrics = dict(
            Loss_cluster_elbo=loss_elbo.item(),
            Loss_cluster_std=loss_std.item(),
            Loss_cluster_embed_total=loss_embed_total.item(),
        )

        # metrics_log = dict(
        #     Cluster_log_data=log_data,
        #     Cluster_kl=kl_from_prior,
        #     Cluster_post_std=posterior.std(dim=-1),
        #     Cluster_std_mean=std_mean,
        # )

        # metrics.update(metrics_log)


        return loss_embed_total, metrics

    
    def assign_cluster_centroids(self, data, space='obs'):
        
        centroids_assigned = data[self.state_key]

        centroids_assigned = torch.Tensor(centroids_assigned)
        
        centroids_assigned = centroids_assigned.view(-1, centroids_assigned.size(-1))

        if space == 'obs':
            # centroids_assigned = random.sample(centroids_assigned, self.config.cluster['n_latent_landmarks'])
            centroids_assigned = self.fps_selection(centroids_assigned, self.config.cluster['n_latent_landmarks'])

            # print(centroids_assigned)

        centroids_assigned = {self.state_key: tf.convert_to_tensor(centroids_assigned)}

        

        # print(centroids_assigned)
        centroids_assigned_embed = self.encoder(centroids_assigned)

        

        
        numpy_embedding = centroids_assigned_embed.numpy()

        torch_tensor_embedding = torch.Tensor(numpy_embedding)

        

        torch_tensor_embedding = torch_tensor_embedding.reshape(-1, self.config.encoder['mlp_layers'][-1])

        if space == 'embed':

            torch_tensor_embedding = self.fps_selection(torch_tensor_embedding, self.config.cluster['n_latent_landmarks'])
            torch_tensor_embedding = torch.Tensor(torch_tensor_embedding)

        

        # print(torch_tensor_embedding.size(0))

        assert type(torch_tensor_embedding) == torch.Tensor and torch_tensor_embedding.size(0) == self.config.cluster['n_latent_landmarks']

        self.cluster.assign_centroids(torch_tensor_embedding)

    
    def fps_selection(
            self,
            goals_embed: torch.Tensor,
            n_select: int,
            inf_value=1e6,
            embed_epsilon=1e-3, early_stop=False,
            embed_op='mean',
    ):
        assert goals_embed.ndim == 2
        n_states = goals_embed.size(0)
        dists = torch.zeros(n_states).to(goals_embed.device) + inf_value
        chosen = []
        while len(chosen) < n_select:
            if dists.max() < embed_epsilon and early_stop:
                break
            idx = dists.argmax()  # farthest point idx
            idx_embed = goals_embed[idx]
            chosen.append(idx)
            # distance from the chosen point to all other pts
            diff_embed = (goals_embed - idx_embed[None, :]).pow(2)
            if embed_op == 'mean':
                new_dists = diff_embed.mean(dim=1)
            elif embed_op == 'sum':
                new_dists = diff_embed.sum(dim=1)
            elif embed_op == 'max':
                new_dists = diff_embed.max(dim=1)[0]
            else:
                raise NotImplementedError
            dists = torch.stack((dists, new_dists.float())).min(dim=0)[0]
        chosen = torch.stack(chosen)
        chosen = chosen.detach().cpu().numpy()

        chosen_goals_embed = goals_embed[chosen]
        return chosen_goals_embed

    def loss(self, data, dynamical_distance_net, state=None):
        # wm_start = time()
        embed = self.encoder(data)
        data['embed'] = tf.cast(tf.stop_gradient(embed), tf.float32) # Needed for the embed head
        post, prior = self.rssm.observe(embed, data['action'], data['is_first'], state) 
        # wm_duration = time() - wm_start
        # print("wm loss1/2 duration", wm_duration)
        # KL loss
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        # scale = 1.0
        # free = self.config.kl['free']
        # balance = self.config.kl['balance']
        # kl_loss, kl_value = self.rssm.lexa_kl_loss(post, prior, balance, free, scale)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)  # feat:(h_t+z_t)

        # image log loss/reward log loss/ discount log loss
        for name, head in self.heads.items():

            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            dists =out if isinstance(out, dict) else {name: out}

            if name == 'decoder':
                obs_decoded = dists[self.state_key].mode()

                if self.obs2goal is not None:
                    obs_decoded_gs = self.obs2goal(obs_decoded)

            for key, dist in dists.items():

                like = tf.cast(dist.log_prob(data[key]), tf.float32)  # log loss
                likes[key] = like
                losses[key] = -like.mean()
        
        if self.config.if_opt_embed_by_dd:
            loss_latent_dd = self.get_loss_latent_dd(data[self.config.state_key], embed, dynamical_distance_net)
            losses['loss_latent_dd'] = loss_latent_dd

        model_loss = sum(self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())  

        outs = dict(
                embed=embed, feat=feat, post=post,
                prior=prior, likes=likes, kl=kl_value)

        if self.obs2goal is not None:
            outs['obs_decoded_gs'] = obs_decoded_gs

        metrics = {f'{name}_loss': value for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    # use dynamical distance net in original code _data_1 = observation, _data_2 = embed, 
    def get_loss_latent_dd(self, _data_1, _data_2, dynamical_distance_net):

        
        

        def _helper(cur_idxs, goal_idxs, distance):

            cur_observation = tf.expand_dims(tf.gather_nd(_data_1, cur_idxs),0)
            goal_observation = tf.expand_dims(tf.gather_nd(_data_1, goal_idxs),0)

            cur_embed = tf.expand_dims(tf.gather_nd(_data_2, cur_idxs),0)
            goal_embed = tf.expand_dims(tf.gather_nd(_data_2, goal_idxs),0)

            # print(L3P_dd_pred)

            # print(cur_states.shape)
            # print(goal_states.shape)
            dd_target_mode = 2

            if dd_target_mode == 0:

                L3P_dd_pred = distance
            
            elif dd_target_mode == 1:

                L3P_dd_pred_1 = tf.cast(self.L3P_dynamical_distance(tf.concat([cur_observation, goal_observation], axis=-1)), tf.float32)
                L3P_dd_pred_2 = tf.cast(self.L3P_dynamical_distance(tf.concat([goal_observation, cur_observation], axis=-1)), tf.float32)
                L3P_dd_pred = tf.stop_gradient(L3P_dd_pred_1 + L3P_dd_pred_2)

            elif dd_target_mode == 2:

                if self.config.gc_input == 'state':

                    cur_goalspace = tf.cast(self.obs2goal(cur_observation), self.dtype)
                    goal_goalspace = tf.cast(self.obs2goal(goal_observation), self.dtype)

                    L3P_dd_pred_1 = tf.cast(dynamical_distance_net(tf.concat([cur_goalspace, goal_goalspace], axis=-1)), tf.float32)
                    L3P_dd_pred_2 = tf.cast(dynamical_distance_net(tf.concat([goal_goalspace, cur_goalspace], axis=-1)), tf.float32)
                    L3P_dd_pred = tf.stop_gradient(L3P_dd_pred_1 + L3P_dd_pred_2)
                    # print(L3P_dd_pred)

                else:
                    L3P_dd_pred_1 = tf.cast(dynamical_distance_net(tf.concat([cur_embed, goal_embed], axis=-1)), tf.float32)
                    L3P_dd_pred_2 = tf.cast(dynamical_distance_net(tf.concat([goal_embed, cur_embed], axis=-1)), tf.float32)
                    L3P_dd_pred = tf.stop_gradient(L3P_dd_pred_1 + L3P_dd_pred_2)
                    # print(L3P_dd_pred)


            latent_dd = tf.reduce_sum(tf.square(cur_embed - goal_embed), axis = -1)

            latent_dd = tf.squeeze(latent_dd)

            latent_dd = tf.cast(latent_dd, tf.float32)

            # print(latent_dd.shape)
            # print(latent_dd)

            loss_latent_dd = tf.square(latent_dd - L3P_dd_pred)

            return loss_latent_dd.mean()
        
        idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self.config.dd_num_positives)
        loss_latent_dd = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])

        return loss_latent_dd

    # use dynamical distance net defined by self _data = observation
    def get_L3P_dd_net_loss(self, _data):

        

        metrics = {}
        bs, seq_len = _data.shape[:2]

        
        

        def _helper(cur_idxs, goal_idxs, distance):
            loss = 0
            cur_states = tf.expand_dims(tf.gather_nd(_data, cur_idxs),0)
            goal_states = tf.expand_dims(tf.gather_nd(_data, goal_idxs),0)
            pred = tf.cast(self.L3P_dynamical_distance(tf.concat([cur_states, goal_states], axis=-1)), tf.float32)
            
            # print(tf.executing_eagerly())
            # tf.print(pred, output_stream=sys.stdout)
            
            if self.config.dd_loss == 'regression':
                _label = distance
                loss += tf.reduce_mean((_label-pred)**2)
            else:
                _label = tf.one_hot(tf.cast(distance, tf.int32), self.dd_out_dim)
                loss += self.dd_loss_fn(_label, pred)
            
            

            return loss
        
        idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self.config.dd_num_positives)
        loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])
        metrics['L3P_dd_net_loss'] = loss

        return loss, metrics

    
    def imagine(self, policy, start, is_terminal, horizon, goal=None):
        if goal is None: # happens when plan2expl actor trains in imag.
            return super().imagine(policy, start, is_terminal, horizon)

        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        goal = flatten(goal)
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.rssm.get_feat(start)
        actor_inp = tf.concat([start['feat'], goal], -1)
        start['action'] = tf.zeros_like(policy(actor_inp).mode())
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            # print(seq['feat'][-1].shape, goal.shape)
            actor_inp = tf.concat([seq['feat'][-1], goal], -1)
            action = policy(tf.stop_gradient(actor_inp)).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value)
        seq = {k: tf.stack(v, 0) for k, v in seq.items()}
        if 'discount' in self.heads:
            disc = self.heads['discount'](seq['feat']).mean()
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
                true_first *= self.config.discount
                disc = tf.concat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
        seq['discount'] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq['weight'] = tf.math.cumprod(
                tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
        return seq

    @tf.function
    def preprocess(self, obs):
        
        obs = obs.copy()
        dtype = prec.global_policy().compute_dtype
        for key, value in obs.items():
            if key.startswith('log_'):
                continue

            
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            
            if value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0 - 0.5
            obs[key] = value
        obs['reward'] = {
                'identity': tf.identity,
                'sign': tf.sign,
                'tanh': tf.tanh,
        }[self.config.clip_rewards](obs['reward'])
        obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
        obs['discount'] *= self.config.discount
        return obs

    
    @tf.function
    def video_pred(self, data, key):
        decoder = self.heads['decoder']
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)

        
        states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
        recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}

        
        prior = self.rssm.imagine(data['action'][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

    
    @tf.function
    def state_pred(self, data):
        key = self.state_key
        decoder = self.heads['decoder']  
        truth = data[key][:6]
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
        recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode()
        return recon, openl, truth

    
    def get_goal(self, obs, training=False, goal_from_env=False, return_goal_gs = False):

        if self.config.gc_input == 'state':

            
            if (not training) or self.config.training_goals == 'env':
                goal = tf.cast(obs[self.goal_key], self.dtype)
                return goal
            
            
            elif self.config.training_goals == 'batch':
                # Use random goals from the same batch
                # This is only run during imagination training
                goal_embed = tf.cast(self.obs2goal(obs[self.state_key]), self.dtype)
                sh = goal_embed.shape
                goal_embed = tf.reshape(goal_embed, (-1, sh[-1]))

                ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
                goal_embed = tf.gather(goal_embed, ids)
                goal_embed = tf.reshape(goal_embed, sh)
                return goal_embed
            
        
        else:

            
            if (not training) or self.config.training_goals == 'env':

                
                # if self.if_L3P_cluster and not goal_from_env:

                #     goal = tf.cast(obs[self.goal_key], self.dtype)
                #     return goal
                
                # Never alter the goal when evaluating
                goal_obs = obs.copy()
                goal_obs[self.state_key] = obs[self.goal_key]
                _embed = self.encoder(goal_obs)
                if self.config.gc_input == 'embed':
                    return _embed
                elif 'feat' in self.config.gc_input:
                    return self.get_init_feat_embed(_embed) if len(_embed.shape) == 2 else tf.vectorized_map(self.get_init_feat_embed, _embed)
            
            
            elif self.config.training_goals == 'batch':

                # demo_obs_batch = next(self.demo_dataset)
                # obs = demo_obs_batch  # imagine only on expert data

                if self.config.train_env_goal_percent > 0:
                    orig_ag_sh = obs[self.state_key].shape
                    num_goals = tf.math.reduce_prod(orig_ag_sh[:-1])
                    num_dgs = tf.cast(tf.cast(num_goals, tf.float32) * self.config.train_env_goal_percent, tf.int32)
                    num_ags = num_goals - num_dgs
                    flat_ags = tf.reshape(obs[self.state_key], (-1, obs[self.state_key].shape[-1]))
                    # flat_dgs = tf.reshape(obs[self.goal_key], (-1, obs[self.goal_key].shape[-1]))
                    ag_ids = tf.random.shuffle(tf.range(tf.shape(flat_ags)[0]))[:num_ags]
                    # dg_ids = tf.random.shuffle(tf.range(tf.shape(flat_dgs)[0]))[:num_dgs]
                    sel_ags = tf.gather(flat_ags, ag_ids)
                    assert self.sample_env_goals is not None, "need to support sample_env_goals"
                    sel_dgs = self.sample_env_goals(num_dgs)
                    all_goals = tf.concat([sel_ags, sel_dgs], 0)
                    goal_embed = self.encoder({self.state_key: all_goals})
                    # shuffle one more time to mix dgs and ags
                    ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
                    goal_embed = tf.gather(goal_embed, ids)
                    goal_embed = tf.reshape(goal_embed, (*orig_ag_sh[:-1], goal_embed.shape[-1]))

                
                else:
                    # Use random goals from the same batch
                    # This is only run during imagination training
                    goal_embed = self.encoder(obs)
                    sh = goal_embed.shape
                    goal_embed = tf.reshape(goal_embed, (-1, sh[-1]))
                    # goal_embed = tf.random.shuffle(goal_embed)    # shuffle doesn't have gradients so need this workaround...
                    ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
                    goal_embed = tf.gather(goal_embed, ids)
                    goal_embed = tf.reshape(goal_embed, sh)

                    if return_goal_gs:
                        obs_goal = obs[self.state_key]
                        obs_goal_sh = obs_goal.shape
                        obs_goal = tf.reshape(obs_goal, (-1, obs_goal_sh[-1]))

                        obs_goal = tf.gather(obs_goal, ids)
                        obs_goal = tf.reshape(obs_goal, obs_goal_sh)

                        assert goal_embed.shape[0] == obs_goal.shape[0] and goal_embed.shape[1] == obs_goal.shape[1]

                if 'feat' in self.config.gc_input:
                    return tf.vectorized_map(self.get_init_feat_embed, goal_embed)
                
                
                else:

                    if return_goal_gs:
                        return goal_embed, obs_goal
                    else:
                        return goal_embed

            elif self.config.training_goals == 'demo_obs':

                # print("obs keys and shapes:")
                # for key, value in obs.items():
                #     print(f"{key}: {value.shape}")

                # print("self.all_demo_trajectories shape")

                # for key, value in self.all_demo_trajectories[0].items():

                #     print(f"{key}: {value.shape}")

                
                batch_size = obs['env_seed'].shape[0]  # 45
                traj_length = obs['env_seed'].shape[1]  # 50
                obs_len = obs[self.state_key].shape[2]  # 39

                def process_goal_obs(env_seeds):
                    batch_size = env_seeds.shape[0]  # 45
                    traj_length = obs['env_seed'].shape[1]  # 50
                    env_seeds = env_seeds.numpy().tolist()

                    
                    new_goals = []
                    for seed in env_seeds:
                        idx = self.seed_list.index(seed)  
                        demo_obs = self.all_demo_trajectories[idx][self.state_key]  # (150, 39)

                        
                        random_idxs = tf.random.uniform(shape=(traj_length,), minval=0, maxval=len(demo_obs), dtype=tf.int32)
                        sampled_goals = tf.gather(demo_obs, random_idxs)  # (50, 39)

                        new_goals.append(sampled_goals)

                    
                    new_goals = tf.stack(new_goals)

                    return new_goals

                
                new_goals = tf.py_function(process_goal_obs, [obs['env_seed'][:, 0]], tf.float32)
                new_goals = tf.ensure_shape(new_goals, (batch_size, traj_length, obs_len))

                goal_obs = obs.copy()
                goal_obs[self.goal_key] = new_goals
                goal_obs[self.state_key] = new_goals
                
                _embed = self.encoder(goal_obs)

                if self.config.gc_input == 'embed':
                    
                    return _embed

                else:

                    raise NotImplementedError



class GCActorCritic(common.Module):

    def __init__(self, config, act_space, tfstep, obs2goal, goal_dim):
        self.config = config
        self.state_key = config.state_key
        self.dtype = prec.global_policy().compute_dtype
        self.act_space = act_space
        self.tfstep = tfstep
        self.obs2goal = obs2goal
        self.goal_dim = goal_dim
        discrete = hasattr(act_space, 'n')

        if self.config.actor.dist == 'auto':
            self.config = self.config.update({'actor.dist': 'onehot' if discrete else 'trunc_normal'})
        if self.config.actor_grad == 'auto':
            self.config = self.config.update({'actor_grad': 'reinforce' if discrete else 'dynamics'})
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)

        # ACH Add
        self.if_train_actor_gs = self.config.if_actor_gs
        if self.if_train_actor_gs:
            self.actor_gs = common.MLP(act_space.shape[0], **self.config.actor)
            self.actor_gs_opt = common.Optimizer('actor', **self.config.actor_opt)

        self.critic = common.MLP([], **self.config.critic)

        if self.config.slow_target:
            self._target_critic = common.MLP([], **self.config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic

        self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
        self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)
        if config.gc_reward == "dynamical_distance":
            dd_out_dim = 1  
            self.dd_loss_fn = tf.keras.losses.MSE  # loss function
            self.dd_seq_len = self.config.imag_horizon
            self.dd_out_dim = dd_out_dim  
            self.dynamical_distance = common.GC_Distance(out_dim = dd_out_dim, input_type= self.config.dd_inp, units=400, normalize_input = self.config.dd_norm_inp)
            self.dd_cur_idxs, self.dd_goal_idxs = get_future_goal_idxs(seq_len = self.config.imag_horizon, bs = self.config.dataset.batch*self.config.dataset.length)
            self._dd_opt = common.Optimizer('dyn_dist', **config.dd_opt)

        # ACH Add
        self.if_reverse_action_converter = self.config.if_reverse_action_converter
        if self.if_reverse_action_converter:

            if self.config.reverse_action_converter.dist == 'auto':
                self.config = self.config.update({'reverse_action_converter.dist': 'onehot' if discrete else 'trunc_normal'})

            self.reverse_action_converter = common.MLP(act_space.shape[0], **self.config.reverse_action_converter)
            self.rac_opt = common.Optimizer('rac', **self.config.actor_opt)

        # ACH Add
        self.if_one_step_predictor = self.config.if_one_step_predictor
        if self.if_one_step_predictor:

            # if self.config.one_step_predictor.dist == 'auto':
            #     self.config = self.config.update({'one_step_predictor.dist': 'onehot' if discrete else 'trunc_normal'})

            self.one_step_predictor = common.MLP(act_space.shape[0], **self.config.actor)
            self.osp_opt = common.Optimizer('osp', **self.config.actor_opt)


        try:

            logdir = pathlib.Path(config.logdir).expanduser()
            demo_replay = common.Replay(logdir / 'demo_episodes', **config.demo_replay)  # initialize replay buffer
            self.demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))

        except:

            print("Can't build demo_dataset in GC_AC")

        self.if_CQL_by_demo = False

        self.if_use_Classifier = False

        if self.if_use_Classifier:

            self.classifier = common.MLP_Classifier(1,  **self.config.classifier)
            self.classifier_opt = common.Optimizer('classifier', **self.config.expl_opt)

            
    
    @tf.function
    def train(self, world_model, start, is_terminal, obs=None):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with tf.GradientTape() as actor_tape:

            

            obs = world_model.preprocess(obs)

            
            
            if self.if_train_actor_gs:
                goal, goal_ori = world_model.get_goal(obs, training=True, return_goal_gs = True) # get goal embeddings from same batch.
            else:
                goal = world_model.get_goal(obs, training=True)

            
            # start is Batch x Length x D.
            seq = world_model.imagine(self.actor, start, is_terminal, hor, goal) # Seq is Horizon x (Batch x Length) x D(h+z).

            # reward = reward_fn(seq)
            imag_feat = seq['feat']
            
            imag_state = seq
            imag_action = seq['action']
            actor_inp = get_actor_inp(imag_feat, goal) # add goal embed to input embedding
            seq['feat_goal'] = actor_inp
            
            reward = self._gc_reward(world_model, actor_inp, imag_state, imag_action, obs)
            seq['reward'], mets1 = self.rewnorm(reward)
            mets1 = {f'reward_{k}': v for k, v in mets1.items()}

            seq['decoded_obs'] = world_model.heads['decoder'](imag_feat)[self.state_key].mode()
            seq['embed'] = world_model.heads['embed'](imag_feat).mode()

            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(world_model, seq, target)

        mets8 = {}
        if self.if_train_actor_gs:  
            with tf.GradientTape() as actor_gs_tape:

                goal_gs = self.obs2goal(goal_ori)
                actor_gs_inp = get_actor_inp(imag_feat, goal_gs)

                actor_gs_loss, mets8 = self.actor_gs_loss(imag_action, actor_gs_inp)

            metrics.update(self.actor_gs_opt(actor_gs_tape, actor_gs_loss, self.actor_gs))

        with tf.GradientTape() as critic_tape:

            if self.if_CQL_by_demo:
                demo_batch_data = next(self.demo_dataset)
                obs = world_model.preprocess(demo_batch_data)
                goal = world_model.get_goal(obs, training=True)

                embed = world_model.encoder(demo_batch_data)
                post, prior = world_model.rssm.observe(embed, demo_batch_data['action'], demo_batch_data['is_first'], state=None)
                feat = world_model.rssm.get_feat(post) 
                feat_goal = tf.concat([feat, goal], -1)

                demo_dist = self.critic(feat_goal[:-1])
                CQL_part = -demo_dist.mode().mean()
                CQL_part = tf.clip_by_value(CQL_part, -20, 20)
                critic_loss, mets4 = self.critic_loss(seq, target, CQL_part)

            else:
                critic_loss, mets4 = self.critic_loss(seq, target)

        mets5 = {}
        if self.config.gc_reward == "dynamical_distance":
            with tf.GradientTape() as df_tape:
                if self.config.gc_input == 'embed':
                    _inp = world_model.heads['embed'](imag_feat).mode()
                elif self.config.gc_input == 'state':
                    _inp = world_model.heads['decoder'](imag_feat)[self.state_key].mode()
                    _inp = tf.cast(self.obs2goal(_inp), self.dtype)
                dd_loss, mets5 = self.get_dynamical_distance_loss(_inp)

            metrics.update(self._dd_opt(df_tape, dd_loss, self.dynamical_distance))

        mets6 = {}
        if self.if_reverse_action_converter:

            with tf.GradientTape() as rac_tape:

                rac_loss, mets6 = self.rac_loss(world_model, obs)

            metrics.update(self.rac_opt(rac_tape, rac_loss, self.reverse_action_converter))

        mets7 = {}
        if self.if_one_step_predictor:

            with tf.GradientTape() as osp_tape:

                osp_loss, mets7 = self.osp_loss(world_model, obs)

            metrics.update(self.osp_opt(osp_tape, osp_loss, self.one_step_predictor))

        mets9 = {}
        if self.if_use_Classifier:
                
            with tf.GradientTape() as classifier_tape:

                classifier_loss, mets9 = self.classifier_loss(world_model, obs)

            metrics.update(self.classifier_opt(classifier_tape, classifier_loss, self.classifier))

        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4, **mets5, **mets6, **mets7, **mets8, **mets9)
        self.update_slow_target()    # Variables exist after first forward pass.
        return metrics


    
    @tf.function
    def BC_train(self, world_model):

        metrics = {}

        with tf.GradientTape() as actor_bc_tape:

            actor_bc_loss, mets = self.bc_loss(world_model)

        metrics.update(mets)
        metrics.update(self.actor_opt(actor_bc_tape, actor_bc_loss, self.actor))
        

        return metrics

    def bc_loss(self, world_model):

        metrics = {}

        demo_batch_data = next(self.demo_dataset)
        obs = world_model.preprocess(demo_batch_data)
        goal = world_model.get_goal(obs)

        embed = world_model.encoder(demo_batch_data)
        post, prior = world_model.rssm.observe(embed, demo_batch_data['action'], demo_batch_data['is_first'], state=None)
        feat = world_model.rssm.get_feat(post)
        feat_goal = tf.concat([feat, goal], -1)

        true_act = demo_batch_data['action']

        feat = feat[:, :-1, :]
        feat_goal = feat_goal[:, :-1, :]
        
        true_act = true_act[:, 1:, :]

        # pred_act_dist = self.actor(feat_goal)
        # # BC_loss = -tf.reduce_mean(pred_act_dist.log_prob(true_act))
        # log_prob = tf.clip_by_value(pred_act_dist.log_prob(true_act), -1e5, 1e5)
        # BC_loss = -tf.reduce_mean(log_prob)

        # metrics['gc_actor_BC_loss'] = BC_loss

        
        pred_act_mean = self.actor(feat_goal).mean()

        
        BC_loss = tf.reduce_mean(tf.square(pred_act_mean - true_act))

        metrics['gc_actor_BC_loss'] = BC_loss
            
        return BC_loss, metrics



    def classifier_loss(self, world_model, obs):

        metrics = {}
        

        if self.config.classifier.input_type == 'obs':

            sampled_obs = obs[self.state_key]

            sampled_obs = tf.reshape(sampled_obs, [-1, tf.shape(sampled_obs)[-1]])
            
            
            demo_batch_data = next(self.demo_dataset)
            demo_data = world_model.preprocess(demo_batch_data)
            demo_obs = demo_data[self.state_key]

            demo_obs = tf.reshape(demo_obs, [-1, tf.shape(demo_obs)[-1]])

            num_samples = 128
            sampled_obs = tf.random.shuffle(sampled_obs)[:num_samples]
            demo_obs = tf.random.shuffle(demo_obs)[:num_samples]

            
            demo_labels = tf.ones([demo_obs.shape[0], 1])       
            non_demo_labels = tf.zeros([sampled_obs.shape[0], 1])  

            
            inputs = tf.concat([demo_obs, sampled_obs], axis=0)
            labels = tf.concat([demo_labels, non_demo_labels], axis=0)

        elif self.config.classifier.input_type == 'embed':

            
            sampled_embed = world_model.encoder(obs)
            sampled_embed = tf.reshape(sampled_embed, [-1, tf.shape(sampled_embed)[-1]])

            
            demo_batch_data = next(self.demo_dataset)
            demo_data = world_model.preprocess(demo_batch_data)
            demo_embed = world_model.encoder(demo_data)
            demo_embed = tf.reshape(demo_embed, [-1, tf.shape(demo_embed)[-1]])


            sampled_labels = tf.zeros([sampled_embed.shape[0], 1])  
            
            demo_labels = tf.ones([demo_embed.shape[0], 1])

            inputs = tf.concat([demo_embed, sampled_embed], axis=0)
            labels = tf.concat([demo_labels, sampled_labels], axis=0)

        
        elif self.config.classifier.input_type == 'feat':

            
            sampled_embed = world_model.encoder(obs)
            post_states, prior_states = world_model.rssm.observe(sampled_embed[:, :], obs['action'][:, :], obs['is_first'][:, :])  # one step imagine
            sampled_feat = world_model.rssm.get_feat(post_states)
            sampled_feat = tf.reshape(sampled_feat, [-1, tf.shape(sampled_feat)[-1]])

            
            demo_batch_data = next(self.demo_dataset)
            demo_data = world_model.preprocess(demo_batch_data)
            demo_embed = world_model.encoder(demo_data)
            post_states, prior_states = world_model.rssm.observe(demo_embed[:, :], demo_data['action'][:, :], demo_data['is_first'][:, :])  # one step imagine
            demo_feat = world_model.rssm.get_feat(post_states)
            demo_feat = tf.reshape(demo_feat, [-1, tf.shape(demo_feat)[-1]])


            sampled_labels = tf.zeros([sampled_feat.shape[0], 1])  
            
            demo_labels = tf.ones([demo_feat.shape[0], 1])

            inputs = tf.concat([demo_feat, sampled_feat], axis=0)
            labels = tf.concat([demo_labels, sampled_labels], axis=0)

        
        pred_probs = self.classifier(inputs)

        
        loss = tf.keras.losses.binary_crossentropy(labels, pred_probs)
        loss = tf.reduce_mean(loss)  

        # l2_lambda = 0.01
        # l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.classifier.trainable_weights]) * l2_lambda

        # loss += l2_loss

        return loss, metrics


    def actor_gs_loss(self, imag_action, actor_gs_inp):

        action_gs_dist = self.actor_gs(actor_gs_inp)

        like = tf.cast(action_gs_dist.log_prob(imag_action), tf.float32)  # log loss

        actor_gs_loss = -like.mean()
        
        return actor_gs_loss, {'actor_gs_loss': actor_gs_loss}
        
    def actor_loss(self, world_model, seq, target):
        # Actions:            0     [a1]    [a2]     a3
        #                                    ^    |    ^    |    ^    |
        #                                 /     v /     v /     v
        # States:         [z0]->[z1]-> z2 -> z3
        # Targets:         t0     [t1]    [t2]
        # Baselines:    [v0]    [v1]     v2        v3
        # Entropies:                [e1]    [e2]
        # Weights:        [ 1]    [w1]     w2        w3
        # Loss:                            l1        l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(tf.stop_gradient(seq['feat_goal'][:-2]))
        if self.config.actor_grad == 'dynamics':
            objective = target[1:]
        elif self.config.actor_grad == 'reinforce':
            baseline = self._target_critic(seq['feat_goal'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            action = tf.stop_gradient(seq['action'][1:-1])
            objective = policy.log_prob(action) * advantage
        elif self.config.actor_grad == 'both':
            baseline = self._target_critic(seq['feat_goal'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq['action'][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics['actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = tf.stop_gradient(seq['weight'])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['actor_ent'] = ent.mean()
        metrics['actor_ent_scale'] = ent_scale

        return actor_loss, metrics

    def critic_loss(self, seq, target, CQL_part = None):
        # States:         [z0]    [z1]    [z2]     z3
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]     v3
        # Weights:        [ 1]    [w1]    [w2]     w3
        # Targets:        [t0]    [t1]    [t2]
        # Loss:                l0        l1        l2
        dist = self.critic(seq['feat_goal'][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq['weight'])

        
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()  # + CQL(self.critic(demo_seq['feat_goal'][:-1]).mode())

        if self.if_CQL_by_demo:

            critic_loss += CQL_part

        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics

    # reverse action converter loss function
    def rac_loss_original(self, world_model, rac_obs):

        obs = rac_obs.copy()

        
        
        shifted_observation = tf.map_fn(lambda x: tf.roll(x, shift=1, axis=0), obs[self.state_key])

        obs['goal'] = shifted_observation

        original_action = obs['action']

        

        goal = world_model.get_goal(obs, training=False) # just use current goal from obs

        
        
        embed = world_model.encoder(obs)
        obs['embed'] = tf.cast(tf.stop_gradient(embed), tf.float32) # Needed for the embed head
        post, prior = world_model.rssm.observe(embed, obs['action'], obs['is_first'], state=None)

        feat = world_model.rssm.get_feat(post)  # feat:(h_t+z_t)

        actor_inp = tf.concat([feat, goal], -1)

        

        reverse_action = tf.stop_gradient(self.actor(actor_inp).mode())

        

        

        output_action_1_dist = self.reverse_action_converter(original_action[:, 1:, :])

        
        like = tf.cast(output_action_1_dist.log_prob(reverse_action[:, 1:, :]), tf.float32)  # log loss
        output_action_1_loss = -like.mean()

        output_action_2_dist = self.reverse_action_converter(reverse_action[:, 1:, :])

        like = tf.cast(output_action_2_dist.log_prob(original_action[:, 1:, :]), tf.float32)  # log loss
        output_action_2_loss = -like.mean()

        rac_loss = output_action_1_loss + output_action_2_loss


        

        # output_action_2 = self.reverse_action_converter(reverse_action)

        

        # output_action_1_loss = tf.reduce_sum(tf.square(output_action_1 - reverse_action), axis = -1)
        # output_action_2_loss = tf.reduce_sum(tf.square(output_action_2 - original_action), axis = -1)
        
        

        # rac_loss = tf.reduce_sum(output_action_1_loss + output_action_2_loss, axis = -1).mean()

        

        return rac_loss, {'rac_loss': rac_loss}

    # reverse action converter loss function
    def rac_loss(self, world_model, rac_obs):

        obs = rac_obs.copy()

        
        
        shifted_observation = tf.map_fn(lambda x: tf.roll(x, shift=1, axis=0), obs[self.state_key])

        obs['goal'] = shifted_observation

        original_action = obs['action']

        osp_inp = tf.concat([obs[self.state_key], obs['goal']], -1)

        

        output_osp_dist = self.one_step_predictor(osp_inp)

        reverse_action = tf.stop_gradient(output_osp_dist.mode())

        

        

        output_action_1_dist = self.reverse_action_converter(original_action[:, 1:, :])

        
        like = tf.cast(output_action_1_dist.log_prob(reverse_action[:, 1:, :]), tf.float32)  # log loss
        output_action_1_loss = -like.mean()

        output_action_2_dist = self.reverse_action_converter(reverse_action[:, 1:, :])

        like = tf.cast(output_action_2_dist.log_prob(original_action[:, 1:, :]), tf.float32)  # log loss
        output_action_2_loss = -like.mean()

        rac_loss = output_action_1_loss + output_action_2_loss

        return rac_loss, {'rac_loss': rac_loss}
    
    # one_step_predictor loss function
    def osp_loss_original(self, world_model, osp_obs):

        obs = osp_obs.copy()

        

        
        shifted_observation = tf.map_fn(lambda x: tf.roll(x, shift=-1, axis=0), obs[self.state_key])

        # label
        shifted_action = tf.map_fn(lambda x: tf.roll(x, shift=-1, axis=0), obs['action'])

        obs['goal'] = shifted_observation

        
        

        goal = world_model.get_goal(obs, training=False) # just use current goal from obs

        
        
        embed = world_model.encoder(obs)
        obs['embed'] = tf.cast(tf.stop_gradient(embed), tf.float32) # Needed for the embed head
        post, prior = world_model.rssm.observe(embed, obs['action'], obs['is_first'], state=None)

        feat = world_model.rssm.get_feat(post)  # feat:(h_t+z_t)

        osp_inp = tf.concat([feat, goal], -1)

        

        osp_inp = osp_inp[:, :-1, :]

        

        output_osp_dist = self.one_step_predictor(osp_inp)

        label = shifted_action[:, :-1, :]

        

        like = tf.cast(output_osp_dist.log_prob(label), tf.float32)  # log loss
        osp_loss = -like.mean()

        

        return osp_loss, {'osp_loss': osp_loss}

    def osp_loss(self, world_model, osp_obs):

        obs = osp_obs.copy()

        

        
        shifted_observation = tf.map_fn(lambda x: tf.roll(x, shift=-1, axis=0), obs[self.state_key])

        # label
        shifted_action = tf.map_fn(lambda x: tf.roll(x, shift=-1, axis=0), obs['action'])

        obs['goal'] = shifted_observation

        
        

        osp_inp = tf.concat([obs[self.state_key], obs['goal']], -1)

        

        osp_inp = osp_inp[:, :-1, :]

        

        output_osp_dist = self.one_step_predictor(osp_inp)

        label = shifted_action[:, :-1, :]

        
        

        like = tf.cast(output_osp_dist.log_prob(label), tf.float32)  # log loss
        osp_loss = -like.mean()

        

        return osp_loss, {'osp_loss': osp_loss}

    def osp_predict_original(self, world_model, osp_obs):

        obs = osp_obs.copy()

        obs = tf.nest.map_structure(tf.tensor, obs)

        for key, value in obs.items():

            obs[key] = tf.expand_dims(value, 0)

        obs = world_model.preprocess(obs)

        

        goal = world_model.get_goal(obs, training=False) # just use current goal from obs

        embed = world_model.encoder(obs)
        obs['embed'] = tf.cast(tf.stop_gradient(embed), tf.float32) # Needed for the embed head
        post, prior = world_model.rssm.observe(embed, obs['action'], obs['is_first'], state=None)

        feat = world_model.rssm.get_feat(post)  # feat:(h_t+z_t)

        osp_inp = tf.concat([feat, goal], -1)

        output_osp_dist = self.one_step_predictor(osp_inp)

        return tf.squeeze(output_osp_dist.mode())
    
    def osp_predict(self, world_model, osp_obs):

        obs = osp_obs.copy()

        obs = tf.nest.map_structure(tf.tensor, obs)

        for key, value in obs.items():

            obs[key] = tf.expand_dims(value, 0)

        obs = world_model.preprocess(obs)

        

        osp_inp = tf.concat([obs[self.state_key], obs['goal']], -1)

        output_osp_dist = self.one_step_predictor(osp_inp)

        return tf.squeeze(output_osp_dist.mode())

    def target(self, seq):
        
        # States:         [z0]    [z1]    [z2]    [z3]
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]    [v3]
        # Discount:     [d0]    [d1]    [d2]     d3
        # Targets:         t0        t1        t2

        metrics = {}

        if self.if_use_Classifier:

            if self.config.classifier.input_type == 'obs':
                demo_obs_prob = self.classifier(seq['decoded_obs'])
            elif self.config.classifier.input_type == 'embed':
                demo_obs_prob = self.classifier(seq['embed'])
            elif self.config.classifier.input_type == 'feat':
                demo_obs_prob = self.classifier(seq['feat'])
            else:
                raise NotImplementedError(self.config.classifier.input_type)
            
            demo_obs_prob = tf.squeeze(demo_obs_prob)
            # demo_obs_prob = tf.clip_by_value(demo_obs_prob, clip_value_min=0.01, clip_value_max=0.99)

            demo_obs_prob_sum = tf.reduce_sum(demo_obs_prob)
            metrics['demo_obs_prob_sum'] = demo_obs_prob_sum                                                              
            reward = tf.cast(seq['reward'], tf.float32) + demo_obs_prob
        
        else:

            reward = tf.cast(seq['reward'], tf.float32)

        # tf.print("==============================")
        # has_inf = tf.reduce_any(tf.math.is_inf(demo_obs_prob))
        # has_inf2 = tf.reduce_any(tf.math.is_inf(reward))
        
        
        
        disc = tf.cast(seq['discount'], tf.float32)
        value = self._target_critic(seq['feat_goal']).mode()
        # Skipping last time step because it is used for bootstrapping.

        target = common.lambda_return(
                reward[:-1], value[:-1], disc[:-1],
                bootstrap=value[-1],
                lambda_=self.config.discount_lambda,
                axis=0)

        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                        self.config.slow_target_fraction)
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)

    
    def get_dynamical_distance_loss(self, _data, corr_factor = None):
        metrics = {}
        seq_len, bs = _data.shape[:2]
        
        # print(seq_len, bs)
        # pred = tf.cast(self.dynamical_distance(tf.concat([_data, _data], axis=-1)), tf.float32)
        # _label = 1.0
        # loss = tf.reduce_mean((_label-pred)**2)
        # return loss, metrics

        def _helper(cur_idxs, goal_idxs, distance):
            loss = 0
            
            # print(cur_idxs)
            # print(goal_idxs)

            cur_states = tf.expand_dims(tf.gather_nd(_data, cur_idxs),0)
            goal_states = tf.expand_dims(tf.gather_nd(_data, goal_idxs),0)

            # print(cur_states, goal_states)


            # ACH Add ===========================================
            


            # equal_idx = tf.cast(equal_idx, self.dtype)[0]
            # mask = equal_idx == 1
            # distance[mask] = 0
            # ==================================================

            
            

            pred = tf.cast(self.dynamical_distance(tf.concat([cur_states, goal_states], axis=-1)), tf.float32)

            if self.config.dd_loss == 'regression':
                _label = distance
                if self.config.dd_norm_reg_label and self.config.dd_distance == 'steps_to_go':
                    _label = _label/self.dd_seq_len
                loss += tf.reduce_mean((_label-pred)**2)
            else:
                _label = tf.one_hot(tf.cast(distance, tf.int32), self.dd_out_dim)
                loss += self.dd_loss_fn(_label, pred)
            return loss

        #positives
        idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self.config.dd_num_positives)
        loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])
        # metrics['dd_pos_loss'] = loss

        #negatives
        corr_factor = corr_factor if corr_factor != None else self.config.dataset.length
        if self.config.dd_neg_sampling_factor>0:
            num_negs = int(self.config.dd_neg_sampling_factor*self.config.dd_num_positives)
            neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
            neg_loss = _helper(neg_cur_idxs, neg_goal_idxs, tf.ones(num_negs)*seq_len)
            loss += neg_loss
            # metrics['dd_neg_loss'] = neg_loss

        return loss, metrics

    
    
    def _gc_reward(self, world_model, feat, inp_state=None, action=None, obs=None):
        # feat is a tensor containing [inp_feat, goal_emb]
        #image embedding as goal
        if self.config.gc_input == 'embed':
            inp_feat, goal_embed = tf.split(feat, [-1, world_model.encoder.embed_size], -1)
            if self.config.gc_reward == 'l2':
                # goal_feat = tf.vectorized_map(self.world_model.get_init_feat_embed, goal_embed)
                # return -tf.reduce_mean((goal_feat - inp_feat) ** 2, -1)
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                goal_state = tf.nest.map_structure(lambda x: flatten(tf.zeros_like(x)), inp_state)
                goal_action = flatten(tf.zeros_like(action))
                is_first = flatten(tf.ones(action.shape[:2], dtype=tf.bool))
                goal_embed = flatten(goal_embed)
                goal_latent, _ = world_model.rssm.obs_step(goal_state, goal_action, goal_embed, is_first, sample=False)
                goal_feat = world_model.rssm.get_feat(goal_latent)
                goal_feat = goal_feat.reshape(inp_feat.shape)
                return -tf.reduce_mean((goal_feat - inp_feat) ** 2, -1)

            elif self.config.gc_reward == 'cosine':
                goal_feat = tf.vectorized_map(world_model.get_init_feat_embed, goal_embed)
                norm = tf.norm(goal_feat, axis =-1)*tf.norm(inp_feat, axis = -1)
                dot_prod = tf.expand_dims(goal_feat,2)@tf.expand_dims(inp_feat,3)
                return tf.squeeze(dot_prod)/(norm+1e-8)

            elif self.config.gc_reward == 'dynamical_distance':
                inp_embed = tf.cast(world_model.heads['embed'](inp_feat).mode(), goal_embed.dtype)
                dd_out = self.dynamical_distance(tf.concat([inp_embed, goal_embed], axis =-1))
                reward = -dd_out
                if self.config.gc_reward_shape == 'sum_diff':
                    # s1 a1 s2 a2 s3
                    # r1 = d(s2) - d(s1)
                    # r2 = d(s3) - d(s2)
                    # r3 = 0, terminal.
                    diff_reward = reward[1:] - reward[:1]
                    reward = tf.concat([diff_reward, tf.zeros_like(diff_reward)[None,0]], 0)
                return reward
        
        elif self.config.gc_input == 'state':
            inp_feat, goal = tf.split(feat, [-1, self.goal_dim], -1)
            if self.config.gc_reward == 'dynamical_distance':
                # need to decode inp_feat to states and then convert to goal space
                # to compare against goal_embed.
                current = tf.cast(world_model.heads['decoder'](inp_feat)[self.state_key].mode(), goal.dtype)
                current = tf.cast(self.obs2goal(current), self.dtype)
                dd_out = self.dynamical_distance(tf.concat([current, goal], axis =-1))
                reward = -dd_out
                if self.config.gc_reward_shape == 'sum_diff':
                    # s1 a1 s2 a2 s3
                    # r1 = d(s2) - d(s1)
                    # r2 = d(s3) - d(s2)
                    # r3 = 0, terminal.
                    diff_reward = reward[1:] - reward[:1]
                    reward = tf.concat([diff_reward, tf.zeros_like(diff_reward)[None,0]], 0)
                return reward
            
            elif self.config.gc_reward == 'l2':
                current = tf.cast(world_model.heads['decoder'](inp_feat)[self.state_key].mode(), goal.dtype)
                current = tf.cast(self.obs2goal(current), self.dtype)
                # TODO: this is block stack specific, abstract out.
                # threshold = 0.05
                # num_blocks = (current.shape[-1] - 5)// 3
                # current_per_obj = tf.concat([current[None, ..., :3], tf.stack(tf.split(current[..., 5:], num_blocks, axis=2))], axis=0)
                # goal_per_obj = tf.concat([goal[None, ..., :3], tf.stack(tf.split(goal[..., 5:], num_blocks, axis=2))], axis=0)
                # dist_per_obj = tf.sqrt(tf.reduce_sum((current_per_obj - goal_per_obj)**2, axis=-1))
                # success_per_obj = tf.cast(dist_per_obj < threshold, self.dtype)
                # grip_success =    success_per_obj[0]
                # obj_success = tf.reduce_prod(success_per_obj[1:], axis=0)
                # reward = 0.1 * grip_success + obj_success

                reward = -tf.reduce_mean((goal - current) ** 2, -1)

                return reward
                # return -tf.reduce_mean((goal - current) ** 2, -1)
            else:
                raise NotImplementedError

    
    def subgoal_dist(self, world_model, obs):
        """Directly converts to embedding with encoder.
        """
        obs = world_model.preprocess(obs)
        if self.config.gc_input == 'embed':
            ob_inp = world_model.encoder(obs)
        elif self.config.gc_input == 'state':
            ob_inp = tf.cast(self.obs2goal(obs[self.state_key]), self.dtype)

        goal_inp = world_model.get_goal(obs, training=False)
        if self.config.gc_reward == 'dynamical_distance':
            dist = self.dynamical_distance(tf.concat([ob_inp, goal_inp], axis =-1))
        elif self.config.gc_reward == 'l2':
            dist = tf.sqrt(tf.reduce_mean((goal_inp - ob_inp) ** 2))
        else:
            raise NotImplementedError
        return dist



class GCActorCritic_Explorer(common.Module):

    def __init__(self, config, act_space, wm, tfstep, obs2goal, goal_dim):
        self.config = config
        self.state_key = config.state_key
        self.dtype = prec.global_policy().compute_dtype
        self.act_space = act_space
        self.wm = wm
        self.tfstep = tfstep
        self.obs2goal = obs2goal
        self.goal_dim = goal_dim
        discrete = hasattr(act_space, 'n')

        if self.config.actor.dist == 'auto':
            self.config = self.config.update({'actor.dist': 'onehot' if discrete else 'trunc_normal'})
        if self.config.actor_grad == 'auto':
            self.config = self.config.update({'actor_grad': 'reinforce' if discrete else 'dynamics'})
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)

        self.critic = common.MLP([], **self.config.critic)

        if self.config.slow_target:
            self._target_critic = common.MLP([], **self.config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic

        self.actor_opt = common.Optimizer('actor_exp', **self.config.actor_opt)
        self.critic_opt = common.Optimizer('critic_exp', **self.config.critic_opt)
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)
        if config.gc_reward == "dynamical_distance":
            dd_out_dim = 1  
            self.dd_loss_fn = tf.keras.losses.MSE  # loss function
            self.dd_seq_len = self.config.imag_horizon
            self.dd_out_dim = dd_out_dim  
            self.dynamical_distance = common.GC_Distance(out_dim = dd_out_dim, input_type= self.config.dd_inp, units=400, normalize_input = self.config.dd_norm_inp)
            self.dd_cur_idxs, self.dd_goal_idxs = get_future_goal_idxs(seq_len = self.config.imag_horizon, bs = self.config.dataset.batch*self.config.dataset.length)
            self._dd_opt = common.Optimizer('dyn_dist_exp', **config.dd_opt)

        # =========================================================== ensemble network ===========================================================
        stoch_size = config.rssm.stoch
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {
                # 'embed': 32 * config.encoder.cnn_depth,
                'embed': wm.encoder.embed_size,
                'stoch': stoch_size,
                'deter': config.rssm.deter,
                'feat': config.rssm.stoch + config.rssm.deter,
                # 'obs_decoded_gs': int(np.prod(wm.shapes[wm.goal_key])),
                'obs_decoded_gs': 7,  # PegInsertaion
        }

        size = size[self.config.disag_target]

        print("explore ensembel ouput size: ", size)

        self.ensemble_networks = [
            common.MLP(size, **config.expl_head)
            for _ in range(config.disag_models)]
        
        self.opt = common.Optimizer('expl_ensemble', **config.expl_opt)
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

        # =========================================================== CQL part ===========================================================
        try:

            logdir = pathlib.Path(config.logdir).expanduser()
            demo_replay = common.Replay(logdir / 'demo_episodes', **config.demo_replay)  # initialize replay buffer
            self.demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))

        except:

            print("Can't build demo_dataset in GC_AC_Explorer")
        
        self.if_CQL_by_demo = False


    
    def train(self, world_model, start, context, is_terminal, obs=None):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with tf.GradientTape() as actor_tape:

            obs = world_model.preprocess(obs)
            goal = world_model.get_goal(obs, training=True)

            # start is Batch x Length x D.
            seq = world_model.imagine(self.actor, start, is_terminal, hor, goal) # Seq is Horizon x (Batch x Length) x D(h+z).

            imag_feat = seq['feat']  # imag_feat.shape: h(200)+z(50)

            imag_state = seq
            imag_action = seq['action']
            actor_inp = get_actor_inp(imag_feat, goal) # add goal embed to input embedding
            seq['feat_goal'] = actor_inp

            # -------------------------------------------------------------------
            start_feat = seq['feat'][0]  
            expanded_start_feat = tf.expand_dims(start_feat, axis=0)  
            expanded_start_feat = tf.repeat(expanded_start_feat, repeats=seq['feat'].shape[0], axis=0)  

            start_feat_goal = tf.concat([expanded_start_feat, actor_inp], axis=-1)
            # -------------------------------------------------------------------

            reward = self._gc_reward(world_model, start_feat_goal, imag_state, imag_action, obs, seq)
            seq['reward'], mets1 = self.rewnorm(reward)
            mets1 = {f'reward_{k}': v for k, v in mets1.items()}

            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(seq, target)

        with tf.GradientTape() as critic_tape:

            if self.if_CQL_by_demo:
                demo_batch_data = next(self.demo_dataset)
                obs = world_model.preprocess(demo_batch_data)
                goal = world_model.get_goal(obs, training=True)

                embed = world_model.encoder(demo_batch_data)
                post, prior = world_model.rssm.observe(embed, demo_batch_data['action'], demo_batch_data['is_first'], state=None)
                feat = world_model.rssm.get_feat(post) 
                feat_goal = tf.concat([feat, goal], -1)

                demo_dist = self.critic(feat_goal[:-1])
                CQL_part = -demo_dist.mode().mean()
                CQL_part = tf.clip_by_value(CQL_part, -20, 20)
                critic_loss, mets4 = self.critic_loss(seq, target, CQL_part)

            else:
                critic_loss, mets4 = self.critic_loss(seq, target)

        mets5 = {}
        if self.config.gc_reward == "dynamical_distance":
            with tf.GradientTape() as df_tape:
                if self.config.gc_input == 'embed':
                    _inp = world_model.heads['embed'](imag_feat).mode()
                elif self.config.gc_input == 'state':
                    _inp = world_model.heads['decoder'](imag_feat)[self.state_key].mode()
                    _inp = tf.cast(self.obs2goal(_inp), self.dtype)
                dd_loss, mets5 = self.get_dynamical_distance_loss(_inp)

            metrics.update(self._dd_opt(df_tape, dd_loss, self.dynamical_distance))

        stoch = start['stoch']
        if self.config.rssm.discrete:
            stoch = tf.reshape(
                    stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
        target = {
                'embed': context['embed'],
                'stoch': stoch,
                'deter': start['deter'],
                'feat': context['feat'],
                'obs_decoded_gs': context.get('obs_decoded_gs', None),
        }[self.config.disag_target]

        inputs = context['feat']
        if self.config.disag_action_cond:
            action = tf.cast(obs['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)
        mets6 = self._train_ensemble(inputs, target)

        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4, **mets5, **mets6)
        self.update_slow_target()    # Variables exist after first forward pass.
        return metrics
    

    def _train_ensemble(self, inputs, targets):

        # print(targets)
        if self.config.disag_offset:
            targets = targets[:, self.config.disag_offset:]
            inputs = inputs[:, :-self.config.disag_offset]
        targets = tf.stop_gradient(targets)
        inputs = tf.stop_gradient(inputs)
        with tf.GradientTape() as tape:
            preds = [head(inputs) for head in self.ensemble_networks]
            preds_mode = [pred.mode() for pred in preds]
            loss = -sum([pred.log_prob(targets).mean() for pred in preds])
        metrics = self.opt(tape, loss, self.ensemble_networks)
        return metrics

    def actor_loss(self, seq, target):
        # Actions:            0     [a1]    [a2]     a3
        #                                    ^    |    ^    |    ^    |
        #                                 /     v /     v /     v
        # States:         [z0]->[z1]-> z2 -> z3
        # Targets:         t0     [t1]    [t2]
        # Baselines:    [v0]    [v1]     v2        v3
        # Entropies:                [e1]    [e2]
        # Weights:        [ 1]    [w1]     w2        w3
        # Loss:                            l1        l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(tf.stop_gradient(seq['feat_goal'][:-2]))
        if self.config.actor_grad == 'dynamics':
            objective = target[1:]
        elif self.config.actor_grad == 'reinforce':
            baseline = self._target_critic(seq['feat_goal'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            action = tf.stop_gradient(seq['action'][1:-1])
            objective = policy.log_prob(action) * advantage
        elif self.config.actor_grad == 'both':
            baseline = self._target_critic(seq['feat_goal'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            objective = policy.log_prob(seq['action'][1:-1]) * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics['actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = tf.stop_gradient(seq['weight'])
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['actor_ent'] = ent.mean()
        metrics['actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target, CQL_part = None):
        # States:         [z0]    [z1]    [z2]     z3
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]     v3
        # Weights:        [ 1]    [w1]    [w2]     w3
        # Targets:        [t0]    [t1]    [t2]
        # Loss:                l0        l1        l2
        dist = self.critic(seq['feat_goal'][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq['weight'])

        
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()  # + CQL(self.critic(demo_seq['feat_goal'][:-1]).mode())

        if self.if_CQL_by_demo:

            critic_loss += CQL_part

        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics

    def target(self, seq):
        # States:         [z0]    [z1]    [z2]    [z3]
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]    [v3]
        # Discount:     [d0]    [d1]    [d2]     d3
        # Targets:         t0        t1        t2
        reward = tf.cast(seq['reward'], tf.float32)
        disc = tf.cast(seq['discount'], tf.float32)
        value = self._target_critic(seq['feat_goal']).mode()
        # Skipping last time step because it is used for bootstrapping.

        
        target = common.lambda_return(
                reward[:-1], value[:-1], disc[:-1],
                bootstrap=value[-1],
                lambda_=self.config.discount_lambda,
                axis=0)
        metrics = {}
        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                        self.config.slow_target_fraction)
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)

    
    def get_dynamical_distance_loss(self, _data, corr_factor = None):
        metrics = {}
        seq_len, bs = _data.shape[:2]
        
        # print(seq_len, bs)
        # pred = tf.cast(self.dynamical_distance(tf.concat([_data, _data], axis=-1)), tf.float32)
        # _label = 1.0
        # loss = tf.reduce_mean((_label-pred)**2)
        # return loss, metrics

        def _helper(cur_idxs, goal_idxs, distance):
            loss = 0
            
            # print(cur_idxs)
            # print(goal_idxs)

            cur_states = tf.expand_dims(tf.gather_nd(_data, cur_idxs),0)
            goal_states = tf.expand_dims(tf.gather_nd(_data, goal_idxs),0)

            # print(cur_states, goal_states)


            # ACH Add ===========================================
            


            # equal_idx = tf.cast(equal_idx, self.dtype)[0]
            # mask = equal_idx == 1
            # distance[mask] = 0
            # ==================================================

            
            

            pred = tf.cast(self.dynamical_distance(tf.concat([cur_states, goal_states], axis=-1)), tf.float32)

            if self.config.dd_loss == 'regression':
                _label = distance
                if self.config.dd_norm_reg_label and self.config.dd_distance == 'steps_to_go':
                    _label = _label/self.dd_seq_len
                loss += tf.reduce_mean((_label-pred)**2)
            else:
                _label = tf.one_hot(tf.cast(distance, tf.int32), self.dd_out_dim)
                loss += self.dd_loss_fn(_label, pred)
            return loss

        #positives
        idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self.config.dd_num_positives)
        loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])
        # metrics['dd_pos_loss'] = loss

        #negatives
        corr_factor = corr_factor if corr_factor != None else self.config.dataset.length
        if self.config.dd_neg_sampling_factor>0:
            num_negs = int(self.config.dd_neg_sampling_factor*self.config.dd_num_positives)
            neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
            neg_loss = _helper(neg_cur_idxs, neg_goal_idxs, tf.ones(num_negs)*seq_len)
            loss += neg_loss
            # metrics['dd_neg_loss'] = neg_loss

        return loss, metrics

    
    

    def _gc_reward(self, world_model, feat, inp_state=None, action=None, obs=None, seq=None):
        # feat is a tensor containing [start_feat, input_feat, goal_embed]

        
        embed_size = world_model.encoder.embed_size  
        total_feat_size = tf.shape(feat)[-1]
        feat_size = (total_feat_size - embed_size) // 2  

        
        start_feat, input_feat, goal_embed = tf.split(feat, [feat_size, feat_size, embed_size], axis=-1)

        reward_type = 0 

        if self.config.gc_input == 'embed':
            
            inp_embed = world_model.heads['embed'](input_feat).mode()  
            inp_embed_cast = tf.cast(inp_embed, goal_embed.dtype)  

            start_embed = world_model.heads['embed'](start_feat).mode()  
            start_embed_cast = tf.cast(start_embed, goal_embed.dtype)  

            
            dd_goal = self.dynamical_distance(tf.concat([inp_embed_cast, goal_embed], axis=-1))
            
            
            dd_start = self.dynamical_distance(tf.concat([inp_embed_cast, start_embed_cast], axis=-1))
            
            
            reward_goal = -dd_goal  
            reward_start = -dd_start  

            reward_intr = self._intr_reward(seq)

            reward_intr = tf.cast(reward_intr, reward_goal.dtype)

            
            reward = reward_goal + reward_intr
            # reward = reward_goal + reward_start + reward_intr

            
            if self.config.gc_reward_shape == 'sum_diff':
                diff_reward = reward[1:] - reward[:-1]  
                reward = tf.concat([diff_reward, tf.zeros_like(diff_reward)[None, 0]], axis=0)

            return reward

        else:
            raise NotImplementedError

    def _intr_reward(self, seq):
        inputs = seq['feat'] # T x B x D
        if self.config.disag_action_cond:
            action = tf.cast(seq['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)
        preds = [head(inputs).mode() for head in self.ensemble_networks]
        disag = tf.tensor(preds).std(0).mean(-1)
        if self.config.disag_log:
            disag = tf.math.log(disag)
        reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]

        
        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.extr_rewnorm(self.reward(seq))[0]

        return reward # T x B



class Goal_optimizer(common.Module):

    def __init__(self, config, tfstep, goal_dim):
        self.config = config
        self.state_key = config.state_key
        self.dtype = prec.global_policy().compute_dtype
        self.tfstep = tfstep
        self.goal_dim = int(goal_dim)

        if self.config.actor.dist == 'auto':
            self.config = self.config.update({'actor.dist': 'trunc_normal'})

        self.actor = common.MLP(self.goal_dim, **self.config.actor)

        self.critic = common.MLP([], **self.config.critic)

        if self.config.slow_target:
            self._target_critic = common.MLP([], **self.config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic

        self.actor_opt = common.Optimizer('goal_opt_actor', **self.config.actor_opt)
        self.critic_opt = common.Optimizer('goal_opt_critic', **self.config.critic_opt)
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)


    
    @tf.function
    def train(self, data=None):

        metrics = {}

        data = self.preprocess(data)

        with tf.GradientTape() as actor_tape:

            target, mets1 = self.target(data)
            actor_loss, mets2 = self.actor_loss(data, target)

        with tf.GradientTape() as critic_tape:

            critic_loss, mets3 = self.critic_loss(data, target)

        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3)
        self.update_slow_target()    # Variables exist after first forward pass.
        return metrics


    def target(self, data):
        # States:         [z0]    [z1]    [z2]    [z3]
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]    [v3]
        # Discount:     [d0]    [d1]    [d2]     d3
        # Targets:         t0        t1        t2
        reward = tf.cast(data['reward'], tf.float32)
        disc = tf.cast(data['discount'], tf.float32)
        value = self._target_critic(data['obs_goal']).mode()
        # Skipping last time step because it is used for bootstrapping.

        target = common.lambda_return(
                reward[:-1], value[:-1], disc[:-1],
                bootstrap=value[-1],
                lambda_=self.config.discount_lambda,
                axis=0)
        
        metrics = {}

        return target, metrics
    

    def actor_loss(self, data, target):

        loss_grad = "reinforce"
        
        metrics = {}

        policy = self.actor(tf.stop_gradient(data['obs_goal'][:-2]))

        if loss_grad == 'reinforce':

            baseline = self._target_critic(data['obs_goal'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            optimized_goal = tf.stop_gradient(data['optimized_goal'][:-2])
            log_prob = tf.clip_by_value(policy.log_prob(optimized_goal), -20, 20)
            objective = log_prob * advantage

        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
        objective += ent_scale * ent

        # weight = tf.stop_gradient(data['weight'])
        # actor_loss = -(weight[:-2] * objective).mean()

        # Remove weight from loss calculation
        actor_loss = -objective.mean()
        
        return actor_loss, metrics


    def critic_loss(self, data, target):
        # States:         [z0]    [z1]    [z2]     z3
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]     v3
        # Weights:        [ 1]    [w1]    [w2]     w3
        # Targets:        [t0]    [t1]    [t2]
        # Loss:                l0        l1        l2
        dist = self.critic(data['obs_goal'][:-1])
        target = tf.stop_gradient(target)

        # weight = tf.stop_gradient(data['weight'])
        # critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()

        # Remove weight from loss calculation
        critic_loss = -dist.log_prob(target).mean()

        metrics = {'goal_optimizer_critic_value': dist.mode().mean()}

        return critic_loss, metrics


    def update_slow_target(self):

        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                        self.config.slow_target_fraction)
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)


    @tf.function
    def give_optimized_goal(self, obs):

        obs = tf.nest.map_structure(tf.tensor, obs)
        obs = self.preprocess(obs)
        actor_input = obs['obs_goal']

        optimized_goal_dist = self.actor(actor_input)

        optimized_goal = optimized_goal_dist.mode()

        return optimized_goal

    @tf.function
    def preprocess(self, obs):
        obs = obs.copy()
        dtype = prec.global_policy().compute_dtype
        for key, value in obs.items():
            if key.startswith('log_'):
                continue

            
            if value.dtype == tf.int32:
                value = value.astype(dtype)
            
            if value.dtype == tf.uint8:
                value = value.astype(dtype) / 255.0 - 0.5
            obs[key] = value
        obs['reward'] = {
                'identity': tf.identity,
                'sign': tf.sign,
                'tanh': tf.tanh,
        }[self.config.clip_rewards](obs['reward'])
        obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
        obs['discount'] *= self.config.discount

        obs['obs_goal'] = tf.concat([obs[self.state_key], obs['goal']], -1)
        
        return obs


def get_future_goal_idxs(seq_len, bs):

        cur_idx_list = []
        goal_idx_list = []
        #generate indices grid
        for cur_idx in range(seq_len):
            for goal_idx in range(cur_idx, seq_len):
                cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
                goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))

        return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, batch_len):
        cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
        goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
        for i in range(num_negs):
            goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
        return cur_idxs, goal_idxs

def get_actor_inp(feat, goal, repeats=None):
    # Image and goal together - input to the actor
    goal = tf.reshape(goal, [1, feat.shape[1], -1])
    goal = tf.repeat(goal, feat.shape[0], 0)
    if repeats:
        goal = tf.repeat(tf.expand_dims(goal, 2), repeats,2)

    if goal.dtype != feat.dtype:
        goal = tf.cast(goal, feat.dtype)

    return tf.concat([feat, goal], -1)


    