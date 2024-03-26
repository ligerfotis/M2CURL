import torch
from gym import spaces
from typing import Dict, Generator, Optional, Union, Any
import numpy as np
import torch as th
from stable_baselines3.common.buffers import (
    RolloutBuffer,
    DictRolloutBuffer,
    ReplayBuffer,
    DictReplayBuffer,
)
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.preprocessing import is_image_space
from torchvision.transforms import transforms
from PIL import Image


def show_stacked_imgs(obs_stack, n_img_channels=3, max_display=16):
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 12))
    n_batch = int(obs_stack.shape[0])
    n_stack = int(obs_stack.shape[1] / n_img_channels)

    for i in range(1, n_stack + 1):
        grid = make_grid(obs_stack[:max_display, (i - 1) * n_img_channels:i * n_img_channels, ...], 4).permute(1, 2,
                                                                                                               0).cpu().numpy()

        fig.add_subplot(1, n_stack, i)
        plt.xticks([])
        plt.yticks([])
        plt.title('Frame: ' + str(i))
        plt.imshow(grid)

    plt.show(block=True)


def apply_image_augmentations(obs, augmentations=None, visualise=False, n_img_channels=1):
    """
    Apply augmentations to image observations using kornia.
    """
    if augmentations is not None:

        # store obs shape and type to convert back to after augmentations
        orig_shape = obs.shape
        orig_type = obs.dtype

        # kornia requires inputs to be in range [0,1]
        # make sure float
        if orig_type == th.uint8:
            obs = obs.to(th.float32)

        # calulate min and max of each image
        obs = obs.view(obs.size(0), obs.size(1), -1)
        batch_min, batch_max = obs.min(axis=2, keepdim=True)[0], obs.max(axis=2, keepdim=True)[0]

        # noramalise to range [0,1] for kornia
        obs = (obs - batch_min) / ((batch_max - batch_min) + 1e-8)

        # convert back to original shape
        obs = obs.view(orig_shape)

        # apply the augmentations
        obs = augmentations(obs)

        # visualise augmentations at this point where range is [0,1]
        if visualise:
            show_stacked_imgs(obs, n_img_channels=n_img_channels)

        # un-normalize back to original input range
        obs = obs.view(obs.size(0), obs.size(1), -1)
        obs = obs * ((batch_max - batch_min) + 1e-8) + batch_min

        # convert back to original shape again
        obs = obs.view(orig_shape)

        if orig_type == th.uint8:
            obs = obs.to(th.uint8)

    return obs


class AugmentedReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            augmentations: th.nn.Sequential = None,
            visualise_aug: bool = False,
            n_stack: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super(AugmentedReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs
        )

        assert is_image_space(self.observation_space), "To apply augmentations the observation space must be an image"

        self.augmentations = augmentations
        self.visualise_aug = visualise_aug
        self.n_img_channels = int(self.observation_space.shape[0] / n_stack)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        # apply augmentations
        obs_aug = apply_image_augmentations(self.to_torch(obs), self.augmentations, self.visualise_aug,
                                            self.n_img_channels)
        next_obs_aug = apply_image_augmentations(self.to_torch(next_obs), self.augmentations, self.visualise_aug,
                                                 self.n_img_channels)

        data = tuple((
            obs_aug,
            self.to_torch(self.actions[batch_inds, 0, :]),
            next_obs_aug,
            self.to_torch(self.dones[batch_inds]),
            self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
        ))

        return ReplayBufferSamples(*data)


class MultiModalAugmentedReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            augmentations: th.nn.Sequential = None,
            augmentations_tactile: th.nn.Sequential = None,
            visualise_aug: bool = False,
            n_stack: int = 1,
            optimize_memory_usage: bool = False,
    ):
        super(MultiModalAugmentedReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs
        )

        assert is_image_space(self.observation_space), "To apply augmentations the observation space must be an image"

        self.augmentations = augmentations
        self.augmentations_tactile = augmentations_tactile
        self.visualise_aug = visualise_aug
        self.n_img_channels = int(self.observation_space.shape[0] / n_stack)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        # apply augmentations
        obs_aug = apply_image_augmentations(self.to_torch(obs), self.augmentations, self.visualise_aug,
                                            self.n_img_channels)
        next_obs_aug = apply_image_augmentations(self.to_torch(next_obs), self.augmentations, self.visualise_aug,
                                                 self.n_img_channels)

        # get grayscale images from original observations
        grayscale_obs = transforms.Grayscale()(self.to_torch(obs))
        grayscale_next_obs = transforms.Grayscale()(self.to_torch(next_obs))

        # concatenate grayscale images with augmented observations across channels
        obs_aug = th.cat((obs_aug, grayscale_obs), dim=1)
        next_obs_aug = th.cat((next_obs_aug, grayscale_next_obs), dim=1)

        data = tuple((
            obs_aug,
            self.to_torch(self.actions[batch_inds, 0, :]),
            next_obs_aug,
            self.to_torch(self.dones[batch_inds]),
            self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
        ))

        return ReplayBufferSamples(*data)


class AugmentedRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            augmentations: th.nn.Sequential = None,
            visualise_aug: bool = False,
            n_stack: int = 1,
    ):
        super(AugmentedRolloutBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs
        )

        assert is_image_space(self.observation_space), "To apply augmentations the observation space must be an image"

        self.augmentations = augmentations
        self.visualise_aug = visualise_aug
        self.n_img_channels = int(self.observation_space.shape[0] / n_stack)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        obs = self.observations[batch_inds]
        obs_aug = apply_image_augmentations(self.to_torch(obs), self.augmentations, self.visualise_aug,
                                            self.n_img_channels)

        data = tuple((
            obs_aug,
            self.to_torch(self.actions[batch_inds]),
            self.to_torch(self.values[batch_inds].flatten()),
            self.to_torch(self.log_probs[batch_inds].flatten()),
            self.to_torch(self.advantages[batch_inds].flatten()),
            self.to_torch(self.returns[batch_inds].flatten()),
        ))
        return RolloutBufferSamples(*data)


class MMAugmentedDictReplayBuffer(DictReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            augmentations: th.nn.Sequential = None,
            augmentations_tactile: th.nn.Sequential = None,
            visualise_aug: bool = False,
            n_stack: int = 1,
    ):
        super(MMAugmentedDictReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs,
                                                          optimize_memory_usage)

        self.augmentations = augmentations
        self.augmentations_tactile = augmentations_tactile
        self.visualise_aug = visualise_aug

        self.n_img_channels = {}
        for key, subspace in self.observation_space.spaces.items():
            if is_image_space(subspace):
                self.n_img_channels[key] = int(subspace.shape[0] / n_stack)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                                   env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        # Convert to PIL Image for each image in the batch and apply the augmentation
        obs = {}
        next_obs = {}
        for key, subspace in self.observation_space.spaces.items():
            # check if key is tactile or visual
            if is_image_space(subspace):
                if key == 'tactile':
                    obs[key] = self.apply_augmentation(obs_[key], self.augmentations_tactile)
                    next_obs[key] = self.apply_augmentation(next_obs_[key], self.augmentations_tactile)
                else:
                    obs[key] = self.apply_augmentation(obs_[key], self.augmentations)
                    next_obs[key] = self.apply_augmentation(next_obs_[key], self.augmentations)
            else:
                obs[key] = obs_[key]
                next_obs[key] = next_obs_[key]

        return DictReplayBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_obs,
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )

    def apply_augmentation(self, images, augmentations=None):
        # Convert the NumPy array to a PIL Image for each image in the batch
        pil_images = [Image.fromarray(image.transpose(1, 2, 0).squeeze().astype(np.uint8)) for image in images]

        # check if images are tactile or visual
        # Apply the augmentation to each PIL Image
        augmented_images = [augmentations(image) for image in pil_images]

        # Convert the augmented PIL Images back to NumPy arrays
        augmented_images = [np.array(image).astype(np.float32) / 255.0 for image in augmented_images]
        augmented_images = np.stack(augmented_images)
        # check if augmented images are tensor. If not, convert them to tensor
        if not torch.is_tensor(augmented_images):
            augmented_images = torch.from_numpy(augmented_images)
        return augmented_images


class AugmentedDictReplayBuffer(DictReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            augmentations: th.nn.Sequential = None,
            visualise_aug: bool = False,
            n_stack: int = 1,
    ):
        super(AugmentedDictReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs,
                                                        optimize_memory_usage)

        self.augmentations = augmentations
        self.visualise_aug = visualise_aug

        self.n_img_channels = {}
        for key, subspace in self.observation_space.spaces.items():
            if is_image_space(subspace):
                self.n_img_channels[key] = int(subspace.shape[0] / n_stack)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:

        obs = {
            key: self.to_torch(self._normalize_obs(obs[batch_inds, 0, :], env)) for key, obs in
            self.observations.items()
        }

        next_obs = {
            key: self.to_torch(self._normalize_obs(obs[batch_inds, 0, :], env)) for key, obs in
            self.next_observations.items()
        }

        # apply augmentations to images
        for key, subspace in self.observation_space.spaces.items():
            if is_image_space(subspace):
                obs[key] = apply_image_augmentations(obs[key], self.augmentations, self.visualise_aug,
                                                     self.n_img_channels[key])
                next_obs[key] = apply_image_augmentations(next_obs[key], self.augmentations, self.visualise_aug,
                                                          self.n_img_channels[key])

        return DictReplayBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[batch_inds]),
            next_observations=next_obs,
            dones=self.to_torch(self.dones[batch_inds]),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
        )


class AugmentedDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
            augmentations: th.nn.Sequential = None,
            visualise_aug: bool = False,
            n_stack: int = 1,
    ):

        super(AugmentedDictRolloutBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs
        )

        self.augmentations = augmentations
        self.visualise_aug = visualise_aug

        self.n_img_channels = {}
        for key, subspace in self.observation_space.spaces.items():
            if is_image_space(subspace):
                self.n_img_channels[key] = int(subspace.shape[0] / n_stack)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictRolloutBufferSamples:

        obs = {
            key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()
        }

        # apply augmentations to images
        for key, subspace in self.observation_space.spaces.items():
            if is_image_space(subspace):
                obs[key] = apply_image_augmentations(obs[key], self.augmentations, self.visualise_aug,
                                                     self.n_img_channels[key])

        return DictRolloutBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )
