

from copy import deepcopy
import tensorflow as tf
from tensorflow_probability import distributions as tfd

import nor_agent
import common
import numpy as np
import pathlib
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Random(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.act_space = act_space

    def actor(self, feat):
        shape = feat.shape[:-1] + self.act_space.shape
        if self.config.actor.dist == 'onehot':
            return common.OneHotDist(tf.zeros(shape))
        else:
            dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
            return tfd.Independent(dist, 1)

    def train(self, start, context, data):
        return None, {}


class Demo_Explorer(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.act_space = act_space

        logdir = pathlib.Path(config.logdir).expanduser()
        demo_replay = common.Replay(logdir / 'demo_episodes', **config.replay)  # initialize replay buffer
        self.demo_dataset = []

        for key, value in demo_replay._complete_eps.items():
            self.demo_dataset.append(value)
        
        self.demo_distance_threshold = 0.01

    # @tf.function
    def find_most_similar_demo_action(self, obs):
        # obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        obs = tf.cast(obs, dtype=tf.float32)

        if len(obs.shape) == len(self.act_space.shape):  
            obs = tf.expand_dims(obs, axis=0)

        batch_size = obs.shape[0]
        best_actions = []

        for i in range(batch_size):
            current_obs = obs[i]
            min_distance = tf.constant(float('inf'))
            best_action = tf.zeros(self.act_space.shape, dtype=tf.float32)

            for demo in self.demo_dataset:
                demo_obs = tf.convert_to_tensor(demo['observation'], dtype=tf.float32)
                demo_act = tf.convert_to_tensor(demo['action'], dtype=tf.float32)

                if demo_obs.shape[0] == 0 or demo_act.shape[0] == 0:
                    continue  

                distances = tf.norm(demo_obs - current_obs, axis=1)
                if tf.reduce_any(tf.math.is_nan(distances)):
                    continue  

                min_index = tf.argmin(distances)
                min_distance_demo = distances[min_index]

                if min_distance_demo < min_distance:
                    min_distance = min_distance_demo
                    best_action = demo_act[tf.minimum(min_index + 1, len(demo_act) - 1)]

            best_actions.append(best_action)

        return tf.stack(best_actions, axis=0), min_distance

    def actor(self, obs):
        # Calculate the shape that matches the obs and action spaces
        shape = obs.shape[:-1] + self.act_space.shape

        # Find the actions for the most similar demo observations for each batch in obs
        demo_actions, min_distance = self.find_most_similar_demo_action(obs)

        # If no similar action is found, default to a uniform distribution
        # if min_distance > self.demo_distance_threshold:
        #     dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
        #     return tfd.Independent(dist, 1)

        # Return the demo_ actions as a deterministic distribution with batch support
        return tfd.Deterministic(tf.convert_to_tensor(demo_actions, dtype=tf.float32))   

    def train(self, start, context, data):
        return None, {}    



class Demo_BC_Explorer(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.act_space = act_space

        try:

            logdir = pathlib.Path(config.logdir).expanduser()
            demo_replay = common.Replay(logdir / 'demo_episodes', **config.demo_replay)  # initialize replay buffer
            self.demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))

        except Exception as e:

            print("Can't build demo_dataset in Demo_BC_Explorer, because of ", e)

        discrete = hasattr(act_space, 'n')
        if self.config.actor.dist == 'auto':
            self.config = self.config.update({'actor.dist': 'onehot' if discrete else 'trunc_normal'})

        explorer_actor_settings = dict(self.config.actor)  
        explorer_actor_settings['min_std'] = 0.0000001
        explorer_actor_settings['max_std'] = 0.0001
        # print(explorer_actor_settings)
        self.actor = common.MLP(act_space.shape[0], **explorer_actor_settings)
        # self.actor = common.MLP(act_space.shape[0], **self.config.actor)
        self.actor_opt = common.Optimizer('demo_bc_actor', **self.config.actor_opt)

        losses = []
        BC_iteration = 1e4
        for i in tqdm(range(int(BC_iteration)), "Pretrain BC actor in Demo_BC_Explorer"):
            
            _, metrics = self.train_bc_actor()
            loss = metrics.get('bc_loss', None)  
            if loss is not None:
                losses.append(loss.numpy())  

        
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='BC Policy Loss')
        plt.title('BC Policy Loss during Pretraining')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        
        plt.savefig(config.logdir + '/bc_policy_loss_plot.png')

        # self.classifier = common.MLP_Classifier(1, **self.config.expl_head)
        # self.classifier_opt = common.Optimizer('classifier', **self.config.expl_opt)

    
    def train(self, start, context, data):
        return None, {}  

    @tf.function
    def train_bc_actor(self):
        metrics = {}

        with tf.GradientTape() as tape:
            loss = self.bc_actor_loss()

        metrics['bc_loss'] = loss

        _ = self.actor_opt(tape, loss, self.actor)

        return None, metrics 


    
    def bc_actor_loss(self):

        demo_batch = next(self.demo_dataset)
        obs = demo_batch['observation'] 
        true_act = demo_batch['action']
        
        obs = obs[:, :-1, :]
        
        true_act = true_act[:, 1:, :]
        
        pred_act_mean = self.actor(obs).mean()

        
        loss = tf.reduce_mean(tf.square(pred_act_mean - true_act))

        return loss
    



class Gail_Explorer(common.Module):
    def __init__(self, config, act_space, wm, tfstep, reward):
        super().__init__()
        self.config = config
        self.reward = reward
        self.wm = wm

        discrete = hasattr(act_space, 'n')
        if self.config.actor.dist == 'auto':
            self.config = self.config.update({'actor.dist': 'onehot' if discrete else 'trunc_normal'})

        # Define actor and critic using common.MLP
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)
        self.critic = common.MLP(1, **self.config.critic)

        self.state_key = config.state_key

        # Initialize replay buffer and demo dataset
        logdir = pathlib.Path(config.logdir).expanduser()
        demo_replay = common.Replay(logdir / 'demo_episodes', **config.replay)
        self.demo_dataset = iter(demo_replay.dataset(**config.dataset))

        # Define classifier and optimizer
        self.classifier = common.MLP_Classifier(1, **self.config.classifier)
        self.classifier_opt = common.Optimizer('classifier', **self.config.expl_opt)

        self.actor_opt = common.Optimizer('gail_actor', **self.config.actor_opt)
        self.critic_opt = common.Optimizer('gail_critic', **self.config.critic_opt)

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def train(self, start, context, data):
        metrics = {}

        # Step 1: Train the classifier (discriminator)
        with tf.GradientTape() as classifier_tape:
            classifier_loss, mets = self.classifier_loss(data)
        metrics.update(self.classifier_opt(classifier_tape, classifier_loss, self.classifier))

        # Step 2: Train the actor and critic
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            rewards = self._intr_reward(data)
            actor_loss, critic_loss = self.actor_critic_loss(data, rewards)

        # Update actor and critic using optimizers
        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))

        return None, metrics

    def classifier_loss(self, obs):
        metrics = {}

        # Get expert and policy samples
        demo_inputs, sampled_inputs = self._prepare_inputs(obs)

        # Create labels
        demo_labels = tf.zeros([tf.shape(demo_inputs)[0], 1])  # Label 0 for demo data
        non_demo_labels = tf.ones([tf.shape(sampled_inputs)[0], 1])  # Label 1 for non-demo data

        # Concatenate inputs and labels
        inputs = tf.concat([demo_inputs, sampled_inputs], axis=0)
        labels = tf.concat([demo_labels, non_demo_labels], axis=0)

        # Predict probabilities using the classifier
        pred_probs = self.classifier(inputs)

        # Binary cross-entropy loss
        loss = -tf.reduce_mean(
            labels * tf.math.log(tf.clip_by_value(pred_probs, 1e-8, 1.0)) +
            (1 - labels) * tf.math.log(tf.clip_by_value(1 - pred_probs, 1e-8, 1.0))
        )

        metrics['classifier_loss'] = loss
        return loss, metrics

    def actor_critic_loss(self, data, rewards):
        """Compute losses for actor and critic with aligned dimensions for rewards and values."""
        features = data[self.state_key]  # Shape: [45, 50, feature_dim]
        actions = data['action']  # Shape: [45, 50, action_dim]
        values = self.critic(features).mode()  # Shape: [45, 50, 1]
        if values.shape[-1] == 1:
            values = tf.squeeze(values, axis=-1)

        # Ensure rewards and values have the same shape
        assert rewards.shape[0] == values.shape[0], "Batch sizes of rewards and values must match"
        assert rewards.shape[1] == values.shape[1], "Time steps of rewards and values must match"

        # Compute returns (discounted cumulative rewards)
        returns = tf.stop_gradient(rewards[:, :-1] + self.config.discount * values[:, 1:])  # Shape: [45, 49]

        # Compute advantages
        advantages = returns - values[:, :-1]  # Shape: [45, 49]
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # Compute log probabilities of the actions
        action_dist = self.actor(features[:, :-1])  # Predict action distributions, Shape: [45, 49, action_dim]
        log_probs = action_dist.log_prob(actions[:, 1:])  # Shape: [45, 49]

        # Actor loss (policy gradient with advantages)
        actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))  # Scalar

        # Critic loss (mean squared error)
        critic_loss = tf.reduce_mean((values[:, :-1] - returns) ** 2)  # Scalar

        return actor_loss, critic_loss


    def _intr_reward(self, data):
        """Compute the GAIL reward based on discriminator (classifier) output."""
        actions = data['action'][1:, :, :]
        actions = tf.cast(actions, tf.float32)

        # Build inputs for the classifier (only using obs)
        observations = data[self.state_key][:-1, :, :]
        observations = tf.cast(observations, tf.float32)

        # Concatenate observations and actions
        input_data = tf.concat([observations, actions], axis=-1)

        # Get discriminator output probabilities
        disc_output = self.classifier(input_data)

        # Compute GAIL reward: negative log-probability
        reward = -tf.math.log(tf.clip_by_value(disc_output, 1e-8, 1.0))
        reward = tf.squeeze(reward, axis=-1)

        # Pad the rewards to match sequence length
        reward = tf.pad(reward, paddings=[[1, 0], [0, 0]], mode='CONSTANT', constant_values=0)

        return reward

    def _prepare_inputs(self, obs):
        """Prepare inputs for the classifier using only obs."""
        # Prepare sampled inputs
        sampled_obs = obs[self.state_key][:, :-1, :]  # Use 'obs' as input
        sampled_actions = obs['action'][:, 1:, :]

        # Reshape observations and actions
        sampled_obs = tf.reshape(sampled_obs, [-1, tf.shape(sampled_obs)[-1]])
        sampled_actions = tf.reshape(sampled_actions, [-1, tf.shape(sampled_actions)[-1]])

        # Convert to float32
        sampled_obs = tf.cast(sampled_obs, tf.float32)
        sampled_actions = tf.cast(sampled_actions, tf.float32)

        # Concatenate observations and actions
        sampled_inputs = tf.concat([sampled_obs, sampled_actions], axis=-1)

        # Prepare demo inputs
        demo_batch_data = next(self.demo_dataset)
        demo_obs = demo_batch_data[self.state_key][:, :-1, :]
        demo_actions = demo_batch_data['action'][:, 1:, :]

        # Reshape demo observations and actions
        demo_obs = tf.reshape(demo_obs, [-1, tf.shape(demo_obs)[-1]])
        demo_actions = tf.reshape(demo_actions, [-1, tf.shape(demo_actions)[-1]])

        # Convert to float32
        demo_obs = tf.cast(demo_obs, tf.float32)
        demo_actions = tf.cast(demo_actions, tf.float32)

        # Concatenate demo observations and actions
        demo_inputs = tf.concat([demo_obs, demo_actions], axis=-1)

        return demo_inputs, sampled_inputs



class Gail_Explorer_wm(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        super().__init__()
        self.config = config
        self.reward = reward
        self.wm = wm
        self.ac = nor_agent.ActorCritic(config, act_space, tfstep)
        self.actor = self.ac.actor
        self.state_key = config.state_key

        logdir = pathlib.Path(config.logdir).expanduser()
        demo_replay = common.Replay(logdir / 'demo_episodes', **config.replay)  # Initialize replay buffer
        self.demo_dataset = iter(demo_replay.dataset(**config.dataset))

        self.classifier = common.MLP_Classifier(1, **self.config.classifier)
        self.classifier_opt = common.Optimizer('classifier', **self.config.expl_opt)

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def train(self, start, context, data):
        metrics = {}

        # Step 1: Train Discriminator
        with tf.GradientTape() as classifier_tape:
            classifier_loss, mets = self.classifier_loss(self.wm, data)
        metrics.update(self.classifier_opt(classifier_tape, classifier_loss, self.classifier))

        # Step 2: Update Policy using Actor-Critic with GAIL reward
        metrics.update(self.ac.train(self.wm, start, data, self._intr_reward))

        return None, metrics

    def classifier_loss(self, world_model, obs):
        metrics = {}

        # Get expert and policy samples
        demo_inputs, sampled_inputs = self._prepare_inputs(world_model, obs)

        # Create labels
        demo_labels = tf.zeros([tf.shape(demo_inputs)[0], 1])       # Label 0 for demo data
        non_demo_labels = tf.ones([tf.shape(sampled_inputs)[0], 1])  # Label 1 for non-demo data

        # Concatenate inputs and labels
        inputs = tf.concat([demo_inputs, sampled_inputs], axis=0)
        labels = tf.concat([demo_labels, non_demo_labels], axis=0)

        # Predict probabilities using the classifier
        pred_probs = self.classifier(inputs)

        # Use log function as part of the discriminator loss
        loss = -tf.reduce_mean(
            labels * tf.math.log(tf.clip_by_value(pred_probs, 1e-8, 1.0)) +
            (1 - labels) * tf.math.log(tf.clip_by_value(1-pred_probs, 1e-8, 1.0))
        )

        metrics['classifier_loss'] = loss
        return loss, metrics

    def _intr_reward(self, seq):
        """Compute the GAIL reward based on discriminator (classifier) output."""
        actions = seq['action'][1:, :, :]
        actions = tf.cast(actions, tf.float32)

        # Build inputs for the classifier
        if self.config.classifier.input_type == 'obs':
            imag_feat = seq['feat']
            seq['decoded_obs'] = self.wm.heads['decoder'](imag_feat)[self.state_key].mode()
            left = seq['decoded_obs'][:-1, :, :]
        elif self.config.classifier.input_type == 'embed':
            left = seq['embed'][:-1, :, :]
        elif self.config.classifier.input_type == 'feat':
            left = seq['feat'][:-1, :, :]
        else:
            raise ValueError(f"Unsupported classifier input_type: {self.config.classifier.input_type}")

        left = tf.cast(left, tf.float32)
        input_data = tf.concat([left, actions], axis=-1)

        # Get discriminator output probabilities
        disc_output = self.classifier(input_data)

        # Compute GAIL reward: negative log-probability
        reward = -tf.math.log(tf.clip_by_value(disc_output, 1e-8, 1.0))
        reward = tf.squeeze(reward, axis=-1)

        # Pad the rewards to match sequence length
        reward = tf.pad(reward, paddings=[[1, 0], [0, 0]], mode='CONSTANT', constant_values=0)

        return reward

    def _prepare_inputs(self, world_model, obs):
        """Prepare inputs for the classifier."""
        if self.config.classifier.input_type == 'obs':
            sampled_obs = obs[self.state_key][:, :-1, :]
            sampled_actions = obs['action'][:, 1:, :]

            sampled_obs = tf.reshape(sampled_obs, [-1, tf.shape(sampled_obs)[-1]])
            sampled_actions = tf.reshape(sampled_actions, [-1, tf.shape(sampled_actions)[-1]])
            sampled_obs = tf.cast(sampled_obs, tf.float32)
            sampled_actions = tf.cast(sampled_actions, tf.float32)
            sampled_inputs = tf.concat([sampled_obs, sampled_actions], axis=-1)

            demo_batch_data = next(self.demo_dataset)
            demo_data = world_model.preprocess(demo_batch_data)
            demo_obs = demo_data[self.state_key][:, :-1, :]
            demo_actions = demo_data['action'][:, 1:, :]

            demo_obs = tf.reshape(demo_obs, [-1, tf.shape(demo_obs)[-1]])
            demo_actions = tf.reshape(demo_actions, [-1, tf.shape(demo_actions)[-1]])
            demo_obs = tf.cast(demo_obs, tf.float32)
            demo_actions = tf.cast(demo_actions, tf.float32)
            demo_inputs = tf.concat([demo_obs, demo_actions], axis=-1)

        elif self.config.classifier.input_type == 'embed':
            sampled_embed = world_model.encoder(obs)[:, :-1, :]
            sampled_actions = obs['action'][:, 1:, :]

            sampled_embed = tf.reshape(sampled_embed, [-1, tf.shape(sampled_embed)[-1]])
            sampled_actions = tf.reshape(sampled_actions, [-1, tf.shape(sampled_actions)[-1]])
            sampled_embed = tf.cast(sampled_embed, tf.float32)
            sampled_actions = tf.cast(sampled_actions, tf.float32)
            sampled_inputs = tf.concat([sampled_embed, sampled_actions], axis=-1)

            demo_batch_data = next(self.demo_dataset)
            demo_data = world_model.preprocess(demo_batch_data)
            demo_embed = world_model.encoder(demo_data)[:, :-1, :]
            demo_actions = demo_data['action'][:, 1:, :]

            demo_embed = tf.reshape(demo_embed, [-1, tf.shape(demo_embed)[-1]])
            demo_actions = tf.reshape(demo_actions, [-1, tf.shape(demo_actions)[-1]])
            demo_embed = tf.cast(demo_embed, tf.float32)
            demo_actions = tf.cast(demo_actions, tf.float32)
            demo_inputs = tf.concat([demo_embed, demo_actions], axis=-1)

        elif self.config.classifier.input_type == 'feat':
            sampled_embed = world_model.encoder(obs)
            post_states, _ = world_model.rssm.observe(sampled_embed, obs['action'], obs['is_first'])
            sampled_feat = world_model.rssm.get_feat(post_states)[:, :-1, :]
            sampled_actions = obs['action'][:, 1:, :]

            sampled_feat = tf.reshape(sampled_feat, [-1, tf.shape(sampled_feat)[-1]])
            sampled_actions = tf.reshape(sampled_actions, [-1, tf.shape(sampled_actions)[-1]])
            sampled_feat = tf.cast(sampled_feat, tf.float32)
            sampled_actions = tf.cast(sampled_actions, tf.float32)
            sampled_inputs = tf.concat([sampled_feat, sampled_actions], axis=-1)

            demo_batch_data = next(self.demo_dataset)
            demo_data = world_model.preprocess(demo_batch_data)
            demo_embed = world_model.encoder(demo_data)
            post_states, _ = world_model.rssm.observe(demo_embed, demo_data['action'], demo_data['is_first'])
            demo_feat = world_model.rssm.get_feat(post_states)[:, :-1, :]
            demo_actions = demo_data['action'][:, 1:, :]

            demo_feat = tf.reshape(demo_feat, [-1, tf.shape(demo_feat)[-1]])
            demo_actions = tf.reshape(demo_actions, [-1, tf.shape(demo_actions)[-1]])
            demo_feat = tf.cast(demo_feat, tf.float32)
            demo_actions = tf.cast(demo_actions, tf.float32)
            demo_inputs = tf.concat([demo_feat, demo_actions], axis=-1)

        return demo_inputs, sampled_inputs



class Plan2Explore(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.reward = reward
        self.wm = wm
        # override certain configs for p2e actor critic.
        p2e_config = deepcopy(config)
        overrides = {
            "discount": config.p2e_discount
        }
        p2e_config = p2e_config.update(overrides)
        self.ac = nor_agent.ActorCritic(p2e_config, act_space, tfstep)  
        self.actor = self.ac.actor
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

        self._networks = [
                common.MLP(size, **config.expl_head)
                for _ in range(config.disag_models)]
        self.opt = common.Optimizer('expl', **config.expl_opt)
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

    def train(self, start, context, data):
        metrics = {}
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
            action = tf.cast(data['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)
        metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self.ac.train(self.wm, start, data, self._intr_reward))

        return None, metrics

    
    def _intr_reward(self, seq):
        inputs = seq['feat'] # T x B x D
        if self.config.disag_action_cond:
            action = tf.cast(seq['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)
        preds = [head(inputs).mode() for head in self._networks]
        disag = tf.tensor(preds).std(0).mean(-1)
        if self.config.disag_log:
            disag = tf.math.log(disag)
        reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]

        
        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.extr_rewnorm(self.reward(seq))[0]

        return reward # T x B

    
    def planner_intr_reward(self, seq):
        # technically second to last timestep since we need (s,a) for p2e rew.
        inputs = feat = seq['feat'] # T x B x D

        
        if self.config.disag_action_cond:
            action = tf.cast(seq['action'], inputs.dtype)
            inputs = tf.concat([inputs, action], -1)

        
        preds = [head(inputs).mode() for head in self._networks]
        disag = tf.tensor(preds).std(0).mean(-1)  

        
        if self.config.disag_log:
            disag = tf.math.log(disag)  

        reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]

        
        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.extr_rewnorm(self.reward(seq))[0]

        
        if self.config.planner.cost_use_p2e_value:
            #discounted sum of rewards plus discounted value of final state.
            # disc = self.config.p2e_discount    * tf.ones(seq['feat'].shape[:-1])
            # accum_disc = tf.math.cumprod(tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
            # sum_rew = (reward * accum_disc)[:-1].sum(0)
            # last_state_value = accum_disc[-1] * self.ac._target_critic(seq['feat'][-1]).mode()
            # returns = sum_rew + last_state_value
            # return returns[None]

            # seq['feat'] is T x B x D
            value = self.ac._target_critic(seq['feat']).mode()
            # value is T x B
            disc = self.config.p2e_discount * tf.ones_like(reward)

            
            returns = common.lambda_return(
                    reward[:-1], value[:-1], disc[:-1],
                    bootstrap=value[-1],
                    lambda_=self.config.discount_lambda,
                    axis=0)
            
            
            if self.config.planner.final_step_cost:
                returns = returns[-10:]

            return returns
        
        else:
            return reward # T x B or 1 x B

    def _train_ensemble(self, inputs, targets):

        # print(targets)
        if self.config.disag_offset:
            targets = targets[:, self.config.disag_offset:]
            inputs = inputs[:, :-self.config.disag_offset]
        targets = tf.stop_gradient(targets)
        inputs = tf.stop_gradient(inputs)
        with tf.GradientTape() as tape:
            preds = [head(inputs) for head in self._networks]
            preds_mode = [pred.mode() for pred in preds]
            loss = -sum([pred.log_prob(targets).mean() for pred in preds])
        metrics = self.opt(tape, loss, self._networks)
        return metrics


# World model prediction error
class ModelLoss(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.reward = reward
        self.wm = wm
        self.ac = nor_agent.ActorCritic(config, act_space, tfstep)
        self.actor = self.ac.actor
        self.head = common.MLP([], **self.config.expl_head)
        self.opt = common.Optimizer('expl', **self.config.expl_opt)

    def train(self, start, context, data):
        metrics = {}
        target = tf.cast(context[self.config.expl_model_loss], tf.float32)

        with tf.GradientTape() as tape:
            loss = -self.head(context['feat']).log_prob(target).mean()

        metrics.update(self.opt(tape, loss, self.head))

        metrics.update(self.ac.train(self.wm, start, data, self._intr_reward))
        
        return None, metrics

    def _intr_reward(self, seq):
        reward = self.config.expl_intr_scale * self.head(seq['feat']).mode()

        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.reward(seq)

        return reward


class RND(common.Module):

    def __init__(self, config, act_space, wm, tfstep, reward):
        self.config = config
        self.reward = reward
        self.wm = wm
        # override certain configs for p2e actor critic.
        p2e_config = deepcopy(config)
        overrides = {
            "discount": config.p2e_discount
        }
        p2e_config = p2e_config.update(overrides)
        self.ac = nor_agent.ActorCritic(p2e_config, act_space, tfstep)
        self.actor = self.ac.actor
        stoch_size = config.rssm.stoch
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {
                # 'embed': 32 * config.encoder.cnn_depth,
                'embed': wm.encoder.embed_size,
                'stoch': stoch_size,
                'deter': config.rssm.deter,
                'feat': config.rssm.stoch + config.rssm.deter,
                'obs_decoded_gs': int(np.prod(wm.shapes[wm.goal_key])),
        }

        size = size[self.config.disag_target]


        # experiment with the network architecuture - 2 or 4
        self._target_network = common.MLP(size, **config.expl_head)
        self._predictor_network = common.MLP(size, **config.expl_head)

        self.opt = common.Optimizer('expl', **config.expl_opt)
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        # self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        self.intr_rewnorm = RunningMeanStd()

    def train(self, start, context, data):
        metrics = {}
        inputs = context['feat']
        _metrics = self._train_predictor(inputs)
        metrics.update(_metrics)
        # tf.print("expl train", self.intr_rewnorm.var)
        metrics.update(self.ac.train(
                self.wm, start, data, self._intr_reward_rnd))
        return None, metrics

    def _intr_reward_rnd(self, seq):
        inputs = seq['feat'] # shape: 16 x 800 x 1224
        # seq['action'] shape = 16 x 800 x 12
        # out size: expl_head = 400 or 512
        # imp: expl_intr_scale ne 1
        f = self._target_network(inputs).mean()
        f_hat = self._predictor_network(inputs).mean()
        reward = self.config.expl_intr_scale * tf.norm(f - f_hat, ord='euclidean', axis=-1)**2
        reward = self.intr_rewnorm.transform(reward)
        # tf.print("_intr_reward_rnd", self.intr_rewnorm.var)
        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.extr_rewnorm(
                    self.reward(seq))[0]
        return reward

    def planner_intr_reward(self, seq):
        # tf.print('planner intr reward', self.intr_rewnorm.var)
        return self._intr_reward_rnd(seq)

    def _train_predictor(self, inputs):
        inputs = tf.stop_gradient(inputs)
        with tf.GradientTape() as tape:
            f = self._target_network(inputs)
            f_hat = self._predictor_network(inputs)
            loss = -f.log_prob(f_hat.mean()).mean()
            reward = self.config.expl_intr_scale * tf.norm(f.mean() - f_hat.mean(), ord='euclidean', axis=-1)**2
            self.intr_rewnorm.update(tf.reshape(reward, [-1]))

        metrics = self.opt(tape, loss, self._predictor_network)
        return metrics


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = tf.Variable(tf.zeros(shape, tf.float32), False)
        self.var = tf.Variable(tf.ones(shape, tf.float32), False)
        self.count = tf.Variable(epsilon, False, dtype=tf.float32)

    def update(self, x):
        batch_mean, batch_var = tf.nn.moments(x, 0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / (tot_count)
        new_var = M2 / (tot_count)

        self.mean.assign(new_mean)
        self.var.assign(new_var)
        self.count.assign(tot_count)

    def transform(self, inputs):
        return inputs / tf.math.sqrt(self.var)