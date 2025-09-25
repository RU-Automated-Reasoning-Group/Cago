import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import register
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS


class MetaWorldSawyerEnv(gym.Env):
    def __init__(self, env_name, seed=False, randomize_hand=False, sparse: bool=False, horizon: int = 250, early_termination: bool=False, width: int=84, height: int=84, generate_image=False):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self.env_name = env_name
        self.env = ALL_V2_ENVIRONMENTS[env_name](render_mode='rgb_array', camera_name='corner2')
        self.env._freeze_rand_vec = False  
        self.env.seeded_rand_vec = True  
        self.env._partially_observable = False
        self.env._set_task_called = True
        self.generate_image = generate_image
        self.width = width
        self.height = height
        self.env.model.vis.global_.offwidth = self.width
        self.env.model.vis.global_.offheight = self.height
        self.env.model.cam_pos
        camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    'corner2',
                )
        self.env.model.cam_pos[camera_id] = np.array([1.5, -0.35, 1.1])
        self.env.model.cam_fovy[camera_id] = 20.0
        
        # self._seed = seed
        # if self._seed:
        #     self.env.seed(0)  # Seed it at zero for now.
        self.randomize_hand = randomize_hand
        self.sparse = sparse
        assert self.env.observation_space.shape[0] == 39
        low, high = self.env.observation_space.low, self.env.observation_space.high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.horizon = horizon
        self._max_episode_steps = min(self.horizon, self.env.max_path_length)
        self.early_termination = early_termination

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self.env.seed(0)

    def step(self, action):
        self._episode_steps += 1
        obs, reward, terminated, truncate, info = self.env.step(action)
        if self._episode_steps == self._max_episode_steps:
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Remove history from the observations. It makes it too hard to reset.
        if self.sparse:
            reward = float(info["success"])  # Reward is now if we succeed or fail.
        else:
            reward = reward / 10
        info['is_success'] = bool(info['success'])
        if self.generate_image:
            info['image'] = self._get_image()
        if self.early_termination and info['success'] > 0:
            terminated = True
        return obs, reward, terminated, truncate, info

    def _get_obs(self):
        return self.env._get_obs()
    
    def _get_image(self):
        return np.fliplr(self.env.render().transpose(2, 0, 1))

    def get_state(self):
        joint_state, mocap_state = self.env.get_env_state()
        qpos, qvel = joint_state.qpos, joint_state.qvel
        mocap_pos, mocap_quat = mocap_state
        self._split_shapes = np.cumsum(
            np.array([qpos.shape[0], qvel.shape[0], mocap_pos.shape[1], mocap_quat.shape[1]])
        )
        return np.concatenate([qpos, qvel, mocap_pos[0], mocap_quat[0], self.env._last_rand_vec], axis=0)

    def set_state(self, state):
        joint_state = self.env.sim.get_state()
        if not hasattr(self, "_split_shapes"):
            self.get_state()  # Load the split
        qpos, qvel, mocap_pos, mocap_quat, rand_vec = np.split(state, self._split_shapes, axis=0)
        if not np.all(self.env._last_rand_vec == rand_vec):
            # We need to set the rand vec and then reset
            self.env._freeze_rand_vec = True
            self.env._last_rand_vec = rand_vec
            self.env.reset()
        joint_state.qpos[:] = qpos
        joint_state.qvel[:] = qvel
        self.env.set_env_state((joint_state, (np.expand_dims(mocap_pos, axis=0), np.expand_dims(mocap_quat, axis=0))))
        self.env.sim.forward()

    def reset(self, **kwargs):
        self._episode_steps = 0
        if "seed" in kwargs:
            self.env.seed(kwargs["seed"])  
        self.env.reset(**kwargs)

        if self.randomize_hand:
            # Hand init pos is usually set to self.init_hand_pos
            # We will add some uniform noise to it.
            high = np.array([0.25, 0.15, 0.2], dtype=np.float32)
            hand_init_pos = self.hand_init_pos + np.random.uniform(low=-high, high=high)
            hand_init_pos = np.clip(hand_init_pos, a_min=self.env.mocap_low, a_max=self.env.mocap_high)
            hand_init_pos = np.expand_dims(hand_init_pos, axis=0)
            mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
            for _ in range(50):
                self.env.data.mocap_pos[mocap_id][:] = self.hand_init_pos
                self.env.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
                self.env.do_simulation([-1, 1], self.env.frame_skip)

        # Get the obs once to reset history.
        self._get_obs()
        return self._get_obs().astype(np.float32), {}

    def render(self):

        image = self.env.render()
        return np.fliplr(np.flipud(self.env.render()))
    

    def __getattr__(self, name):
        return getattr(self.env, name)


class MetaWorldSawyerImageEnv(gym.Env):
    def __init__(self, env_name, seed=False, randomize_hand=True, sparse: bool=False, horizon: int = 250, early_termination: bool=False, width: int=84, height: int=84):
        from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

        self.env_name = env_name
        self.env = ALL_V2_ENVIRONMENTS[env_name](render_mode='rgb_array', camera_name='corner2')
        self.env._freeze_rand_vec = False
        self.env._partially_observable = False
        self.env._set_task_called = True
        self.width = width
        self.height = height
        self.env.model.vis.global_.offwidth = self.width
        self.env.model.vis.global_.offheight = self.height
        self.env.model.cam_pos
        camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    'corner2',
                )
        self.env.model.cam_pos[camera_id] = np.array([1.5, -0.35, 1.1])
        self.env.model.cam_fovy[camera_id] = 20.0
        self._seed = seed
        if self._seed:
            self.env.seed(0)  # Seed it at zero for now.
        self.randomize_hand = randomize_hand
        self.sparse = sparse
        shape = (3, self.width, self.height)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        self.action_space = self.env.action_space
        self.horizon = horizon
        self._max_episode_steps = min(self.horizon, self.env.max_path_length)
        self.early_termination = early_termination

    def seed(self, seed=None):
        super().seed(seed=seed)
        if self._seed:
            self.env.seed(0)

    def step(self, action):
        self._episode_steps += 1
        state_obs, reward, terminated, truncate, info = self.env.step(action)
        info['state'] = state_obs
        if self._episode_steps == self._max_episode_steps:
            info["discount"] = 1.0  # Ensure infinite boostrap.
        # Remove history from the observations. It makes it too hard to reset.
        if self.sparse:
            reward = float(info["success"])  # Reward is now if we succeed or fail.
        else:
            reward = reward / 10
        info['is_success'] = bool(info['success'])
        if self.early_termination and info['success'] > 0:
            terminated = True
        return self._get_image(), reward, terminated, truncate, info

    def _get_obs(self):
        return self.env._get_obs()

    def _get_image(self):
        return np.fliplr(self.env.render().transpose(2, 0, 1))

    def get_state(self):
        joint_state, mocap_state = self.env.get_env_state()
        qpos, qvel = joint_state.qpos, joint_state.qvel
        mocap_pos, mocap_quat = mocap_state
        self._split_shapes = np.cumsum(
            np.array([qpos.shape[0], qvel.shape[0], mocap_pos.shape[1], mocap_quat.shape[1]])
        )
        return np.concatenate([qpos, qvel, mocap_pos[0], mocap_quat[0], self.env._last_rand_vec], axis=0)

    def set_state(self, state):
        joint_state = self.env.sim.get_state()
        if not hasattr(self, "_split_shapes"):
            self.get_state()  # Load the split
        qpos, qvel, mocap_pos, mocap_quat, rand_vec = np.split(state, self._split_shapes, axis=0)
        if not np.all(self.env._last_rand_vec == rand_vec):
            # We need to set the rand vec and then reset
            self.env._freeze_rand_vec = True
            self.env._last_rand_vec = rand_vec
            self.env.reset()
        joint_state.qpos[:] = qpos
        joint_state.qvel[:] = qvel
        self.env.set_env_state((joint_state, (np.expand_dims(mocap_pos, axis=0), np.expand_dims(mocap_quat, axis=0))))
        self.env.sim.forward()

    def reset(self, **kwargs):

        self._episode_steps = 0
        self.env.reset(**kwargs)
        if self.randomize_hand:
            # Hand init pos is usually set to self.init_hand_pos
            # We will add some uniform noise to it.
            high = np.array([0.25, 0.15, 0.2], dtype=np.float32)
            hand_init_pos = self.hand_init_pos + np.random.uniform(low=-high, high=high)
            hand_init_pos = np.clip(hand_init_pos, a_min=self.env.mocap_low, a_max=self.env.mocap_high)
            hand_init_pos = np.expand_dims(hand_init_pos, axis=0)
            mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
            for _ in range(50):
                self.env.data.mocap_pos[mocap_id][:] = self.hand_init_pos
                self.env.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
                self.env.do_simulation([-1, 1], self.env.frame_skip)

        return self._get_image(), {}

    def __getattr__(self, name):
        return getattr(self.env, name)



