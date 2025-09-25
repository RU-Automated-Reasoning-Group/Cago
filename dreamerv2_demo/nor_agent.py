

import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import explorer
import pathlib
from tqdm import tqdm
import tensorflow_probability as tfp
import matplotlib.pyplot as plt



class Agent(common.Module):

    def __init__(self, config, obs_space, act_space, step):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.tfstep = tf.Variable(int(self.step), tf.int64)

        
        self.wm = WorldModel(config, obs_space, self.tfstep)

        
        self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)

        
        if config.expl_behavior == 'greedy':
            self._expl_behavior = self._task_behavior
        else:
            self._expl_behavior = getattr(explorer, config.expl_behavior)(
                    self.config, self.act_space, self.wm, self.tfstep,
                    lambda seq: self.wm.heads['reward'](seq['feat']).mode())


    
    
    @tf.function
    def policy(self, obs, state=None, mode='train'):
        obs = tf.nest.map_structure(tf.tensor, obs)
        tf.py_function(lambda: self.tfstep.assign(int(self.step), read_value=False), [], [])

        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = tf.zeros((len(obs['reward']),) + self.act_space.shape)

            state = latent, action

        
        latent, action = state

        
        embed = self.wm.encoder(self.wm.preprocess(obs))

        sample = (mode == 'train') or not self.config.eval_state_mean

        
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample)

        feat = self.wm.rssm.get_feat(latent)  


        
        if mode == 'eval':
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise

        elif mode == 'explore':
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise

        elif mode == 'train':
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise

        
        action = common.action_noise(action, noise, self.act_space)
        outputs = {'action': action}
        state = (latent, action)
        return outputs, state

    
    @tf.function
    def train(self, data, state=None):
        metrics = {}

        
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs['post']
        reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()

        
        metrics.update(self._task_behavior.train(self.wm, start, data['is_terminal'], reward))

        if self.config.expl_behavior != 'greedy':
            mets = self._expl_behavior.train(start, outputs, data)[-1]
            metrics.update({'expl_' + key: value for key, value in mets.items()})

        return state, metrics

    
    @tf.function
    def report(self, data, env):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads['decoder'].cnn_keys:
            name = key.replace('/', '_')
            
            report[f'openl_{name}'] = self.wm.video_pred(data, key)
        return report

    def agent_save(self, logdir):

        self.save(logdir / 'variables.pkl')

    def agent_load(self, logdir):

        self.load(logdir / 'variables.pkl')


class WorldModel(common.Module):

    def __init__(self, config, obs_space, tfstep):
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.state_key = config.state_key
        self.tfstep = tfstep
        self.rssm = common.EnsembleRSSM(**config.rssm)
        self.encoder = common.Encoder(shapes, self.state_key, **config.encoder)
        self.embed_size = self.encoder.embed_size
        self.heads = {}
        self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
        self.heads['reward'] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads['discount'] = common.MLP([], **config.discount_head)
        if config.pred_embed:
            self.heads['embed'] = common.MLP([self.embed_size], **config.embed_head)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.Optimizer('model', **config.model_opt)


    
    def train(self, data, state=None):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)  
        modules = [self.encoder, self.rssm, *self.heads.values()]
        metrics.update(self.model_opt(model_tape, model_loss, modules))  
        return state, outputs, metrics


    
    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        data['embed'] = tf.cast(tf.stop_gradient(embed), tf.float32) # Needed for the embed head
        post, prior = self.rssm.observe(embed, data['action'], data['is_first'], state)
        
        
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)

        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)

        
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else tf.stop_gradient(feat)
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = tf.cast(dist.log_prob(data[key]), tf.float32)
                likes[key] = like
                losses[key] = -like.mean()

        
        model_loss = sum(self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        
        outs = dict(
                embed=embed, feat=feat, post=post,
                prior=prior, likes=likes, kl=kl_value)
        
        metrics = {f'{name}_loss': value for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        last_state = {k: v[:, -1] for k, v in post.items()}

        return model_loss, last_state, outs, metrics


    
    def imagine(self, policy, start, is_terminal, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.rssm.get_feat(start)
        start['action'] = tf.zeros_like(policy(start['feat']).mode())
        seq = {k: [v] for k, v in start.items()}

        
        for _ in range(horizon):
            action = policy(tf.stop_gradient(seq['feat'][-1])).sample()
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
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()

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



class ActorCritic(common.Module):

    def __init__(self, config, act_space, tfstep):
        self.config = config
        self.state_key = config.state_key
        self.act_space = act_space
        self.tfstep = tfstep
        discrete = hasattr(act_space, 'n')
        if self.config.actor.dist == 'auto':
            self.config = self.config.update({
                    'actor.dist': 'onehot' if discrete else 'trunc_normal'})
        if self.config.actor_grad == 'auto':
            self.config = self.config.update({
                    'actor_grad': 'reinforce' if discrete else 'dynamics'})
        self.actor = common.MLP(act_space.shape[0], **self.config.actor)
        self.critic = common.MLP([], **self.config.critic)
        if self.config.slow_target:
            self._target_critic = common.MLP([], **self.config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
        self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
        self.rewnorm = common.StreamNorm(**self.config.reward_norm)

        self.if_CQL_by_demo = False

        self.if_use_Classifier = False

        self.if_use_bc_KL_constraint = config.if_use_bc_KL_constraint


        try:

            logdir = pathlib.Path(config.logdir).expanduser()
            demo_replay = common.Replay(logdir / 'demo_episodes', **config.demo_replay)  # initialize replay buffer
            self.demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))

        except Exception as e:

            print("Can't build demo_dataset in Nor_AC, because of ", e)

        if self.if_use_Classifier:
            self.classifier = common.MLP_Classifier(1,  **self.config.classifier)
            self.classifier_opt = common.Optimizer('nor_classifier', **self.config.expl_opt)

        if self.if_use_bc_KL_constraint:

            explorer_actor_settings = dict(self.config.actor)  
            explorer_actor_settings['min_std'] = 0.0000001
            explorer_actor_settings['max_std'] = 0.0001

            self.bc_actor = common.MLP(act_space.shape[0], **explorer_actor_settings)
            self.bc_actor_opt = common.Optimizer('demo_bc_actor', **self.config.actor_opt)

            losses = []
            BC_iteration = 1e4
            for i in tqdm(range(int(BC_iteration)), "Pretrain BC policy in P2E"):
                
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



    
    @tf.function
    def train(self, world_model, start, is_terminal, reward_fn):

        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.

        
        with tf.GradientTape() as actor_tape:
            seq = world_model.imagine(self.actor, start, is_terminal, hor)
            reward = reward_fn(seq)
            seq['reward'], mets1 = self.rewnorm(reward)
            mets1 = {f'reward_{k}': v for k, v in mets1.items()}

            imag_feat = seq['feat']
            seq['decoded_obs'] = world_model.heads['decoder'](imag_feat)[self.state_key].mode()
            seq['embed'] = world_model.heads['embed'](imag_feat).mode()
            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(seq, target)

        
        with tf.GradientTape() as critic_tape:
            if self.if_CQL_by_demo:
                demo_batch_data = next(self.demo_dataset)
                obs = world_model.preprocess(demo_batch_data)
                embed = world_model.encoder(demo_batch_data)
                post, prior = world_model.rssm.observe(embed, demo_batch_data['action'], demo_batch_data['is_first'], state=None)
                feat = world_model.rssm.get_feat(post) 
                demo_dist = self.critic(feat[:-1])
                CQL_part = -demo_dist.mode().mean()

                CQL_part = tf.clip_by_value(CQL_part, -20, 20)
                critic_loss, mets4 = self.critic_loss(seq, target, CQL_part)

            else:
                critic_loss, mets4 = self.critic_loss(seq, target)


        mets5 = {}
        if self.if_use_Classifier:
                
            with tf.GradientTape() as classifier_tape:

                classifier_loss, mets5 = self.classifier_loss(world_model, ) 

            metrics.update(self.classifier_opt(classifier_tape, classifier_loss, self.classifier))
    

        
        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()    # Variables exist after first forward pass.

        return metrics

    @tf.function
    def train_bc_actor(self):
        metrics = {}

        with tf.GradientTape() as tape:
            loss = self.bc_actor_loss()

        metrics['bc_loss'] = loss

        _ = self.bc_actor_opt(tape, loss, self.bc_actor)

        return None, metrics 

    
    def bc_actor_loss(self):

        demo_batch = next(self.demo_dataset)
        obs = demo_batch[self.config.state_key] 
        true_act = demo_batch['action']
        
        obs = obs[:, :-1, :]
        
        true_act = true_act[:, 1:, :]
        
        pred_act_mean = self.bc_actor(obs).mean()

        
        loss = tf.reduce_mean(tf.square(pred_act_mean - true_act))

        return loss
    

    
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

        embed = world_model.encoder(demo_batch_data)
        post, prior = world_model.rssm.observe(embed, demo_batch_data['action'], demo_batch_data['is_first'], state=None)
        feat = world_model.rssm.get_feat(post) 

        true_act = demo_batch_data['action']

        feat = feat[:, :-1, :]
        
        true_act = true_act[:, 1:, :]

        # pred_act_dist = self.actor(feat)
        # # BC_loss = -tf.reduce_mean(pred_act_dist.log_prob(true_act))
        # log_prob = tf.clip_by_value(pred_act_dist.log_prob(true_act), -1e5, 1e5)
        # BC_loss = -tf.reduce_mean(log_prob)

        # metrics['nor_actor_BC_loss'] = BC_loss

        
        pred_act_mean = self.actor(feat).mean()

        
        BC_loss = tf.reduce_mean(tf.square(pred_act_mean - true_act))

        metrics['nor_actor_BC_loss'] = BC_loss

        
        return BC_loss, metrics


    
    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(tf.stop_gradient(seq['feat'][:-2]))
        if self.config.actor_grad == 'dynamics':
            objective = target[1:]
        elif self.config.actor_grad == 'reinforce':
            baseline = self._target_critic(seq['feat'][:-2]).mode()
            advantage = tf.stop_gradient(target[1:] - baseline)
            action = tf.stop_gradient(seq['action'][1:-1])
            objective = policy.log_prob(action) * advantage
        elif self.config.actor_grad == 'both':
            baseline = self._target_critic(seq['feat'][:-2]).mode()
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

        if self.if_use_bc_KL_constraint:
            # Calculate KL divergence between the RL policy and Behavior Clone policy
            bc_policy = self.bc_actor(seq['decoded_obs'][:-2])
            kl_div = kl_divergence_truncated_normal(policy, bc_policy)
            kl_constraint_loss = tf.reduce_mean(kl_div)
            
            # Introduce a scaling coefficient for the KL constraint loss
            kl_coeff = 1
            metrics['kl_constraint_loss'] = kl_constraint_loss
            metrics['kl_coeff'] = kl_coeff
            metrics['original_actor_loss'] = actor_loss

            # Add KL constraint loss to actor loss
            actor_loss += kl_coeff * kl_constraint_loss

        return actor_loss, metrics


    
    def critic_loss(self, seq, target, CQL_part=0):
        # States:         [z0]    [z1]    [z2]     z3
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]     v3
        # Weights:        [ 1]    [w1]    [w2]     w3
        # Targets:        [t0]    [t1]    [t2]
        # Loss:                l0        l1        l2
        dist = self.critic(seq['feat'][:-1])
        target = tf.stop_gradient(target)
        weight = tf.stop_gradient(seq['weight'])
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()

        if self.if_CQL_by_demo:
            critic_loss += CQL_part
            
        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics
    
    
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
    

    
    def target(self, seq):
        # States:         [z0]    [z1]    [z2]    [z3]
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]    [v3]
        # Discount:       [d0]    [d1]    [d2]     d3
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

        disc = tf.cast(seq['discount'], tf.float32)
        value = self._target_critic(seq['feat']).mode()
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
                mix = 1.0 if self._updates == 0 else float(self.config.slow_target_fraction)  

                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            
            self._updates.assign_add(1)




def kl_divergence_truncated_normal(policy, bc_policy):
    # Access the underlying distribution for 'Independent' objects
    mu_p, sigma_p = policy.distribution.loc, policy.distribution.scale
    mu_q, sigma_q = bc_policy.distribution.loc, bc_policy.distribution.scale
    
    kl_div = tf.math.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5
    return tf.reduce_mean(kl_div)



