# create a class clalled RAD_SAC that inherits from SAC
# add augmentations and visualise_aug to the __init__ function
import io
import pathlib
from typing import Union, Type, Tuple, Optional, Dict, Any, Iterable

import numpy as np

from sb3_contrib.rad.rad_ppo import RAD_PPO
from stable_baselines3.common.preprocessing import preprocess_obs
from torch.nn import functional as F

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.sac.policies import SACPolicy
import torch as th
from torchvision.transforms import transforms

from augmented_buffers import AugmentedDictReplayBuffer, AugmentedReplayBuffer, MultiModalAugmentedReplayBuffer, \
    MMAugmentedDictReplayBuffer
from sb3_contrib import RAD_SAC


#
# class RAD_SAC(SAC):
#     def __init__(
#             self,
#             policy: Union[str, Type[SACPolicy]],
#             env: Union[GymEnv, str],
#             learning_rate: Union[float, Schedule] = 3e-4,
#             buffer_size: int = 1000000,
#             learning_starts: int = 100,
#             batch_size: int = 256,
#             tau: float = 0.005,
#             gamma: float = 0.99,
#             train_freq: Union[int, Tuple[int, str]] = 1,
#             gradient_steps: int = 1,
#             action_noise: Optional[ActionNoise] = None,
#             optimize_memory_usage: bool = False,
#             ent_coef: Union[str, float] = "auto",
#             target_update_interval: int = 1,
#             target_entropy: Union[str, float] = "auto",
#             use_sde: bool = False,
#             sde_sample_freq: int = -1,
#             use_sde_at_warmup: bool = False,
#             tensorboard_log: Optional[str] = None,
#             create_eval_env: bool = False,
#             policy_kwargs: Dict[str, Any] = None,
#             verbose: int = 0,
#             seed: Optional[int] = None,
#             device: Union[th.device, str] = "auto",
#             _init_setup_model: bool = True,
#             augmentations: th.nn.Sequential = None,
#             visualise_aug: bool = False,
#     ):
#
#         self.augmentations = augmentations
#         self.visualise_aug = visualise_aug
#
#         super(RAD_SAC, self).__init__(
#             policy,
#             env,
#             learning_rate,
#             buffer_size,
#             learning_starts,
#             batch_size,
#             tau,
#             gamma,
#             train_freq,
#             gradient_steps,
#             action_noise,
#             optimize_memory_usage=optimize_memory_usage,
#             ent_coef=ent_coef,
#             target_update_interval=target_update_interval,
#             target_entropy=target_entropy,
#             use_sde=use_sde,
#             sde_sample_freq=sde_sample_freq,
#             use_sde_at_warmup=use_sde_at_warmup,
#             tensorboard_log=tensorboard_log,
#             policy_kwargs=policy_kwargs,
#             verbose=verbose,
#             seed=seed,
#             device=device,
#             _init_setup_model=_init_setup_model,
#         )
#
#     def _setup_model(self) -> None:
#         super(RAD_SAC, self)._setup_model()
#
#         # create a buffer for the augmented observations
#         buffer_cls = AugmentedDictReplayBuffer if isinstance(self.observation_space,
#                                                              gym.spaces.Dict) else AugmentedReplayBuffer
#
#         try:
#             n_stack = self.env.n_stack
#         except:
#             n_stack = 1
#
#         self.replay_buffer = buffer_cls(
#             self.buffer_size,
#             self.observation_space,
#             self.action_space,
#             self.device,
#             augmentations=self.augmentations,
#             visualise_aug=self.visualise_aug,
#             n_stack=n_stack,
#             n_envs=self.n_envs,
#             optimize_memory_usage=self.optimize_memory_usage,
#         )
#
#     def save(
#             self,
#             path: Union[str, pathlib.Path, io.BufferedIOBase],
#             exclude: Optional[Iterable[str]] = None,
#             include: Optional[Iterable[str]] = None,
#     ) -> None:
#         super(RAD_SAC, self).save(
#             path,
#             exclude=['augmentations', 'visualise_aug']
#         )
#

class CURL_SAC(RAD_SAC):
    def __init__(
            self,
            *args,
            curl_alpha: float = 0.1,  # Coefficient for CURL loss
            curl_tau: float = 0.05,  # Soft update coefficient for CURL
            **kwargs
    ):
        super(CURL_SAC, self).__init__(*args, **kwargs)
        self.curl_alpha = curl_alpha
        self.curl_tau = curl_tau

    def train(self, gradient_steps: int, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train(gradient_steps, batch_size)

        # Train CURL
        if self.augmentations is not None:
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = replay_data.observations

            # with th.no_grad():
            #     target_obs = self.replay_buffer.augment_obs(obs.clone()).detach()
            # preprocess the observations
            preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=True)
            # compute the infoNCE loss
            infoNCE_loss = self.policy.actor.features_extractor.compute_loss(preprocessed_obs, self.curl_tau)

            # Backpropagate CURL loss and update the online encoder
            self.policy.actor.optimizer.zero_grad()
            infoNCE_loss.backward()
            self.policy.actor.optimizer.step()

            # Soft update of the momentum encoder
            self.actor.features_extractor.update_momentum_encoder(self.curl_tau)

            self.logger.record("train/infoNCE_loss", infoNCE_loss.item())


class MViTacRL_SAC(RAD_SAC):
    def __init__(
            self,
            *args,
            beta: float = 0.1,  # Coefficient for CURL loss
            tau: float = 0.05,  # Soft update coefficient for CURL
            lambda_vis: float = 1,
            lambda_tac: float = 1,
            lambda_vis_tac: float = 1,
            # augmentations: Dict[str, th.nn.Sequential] = None,
            **kwargs
    ):
        # self.augmentations = augmentations
        # self.augmentations = augmentations["visual"]
        # self.augmentations_tactile = augmentations["tactile"]
        # kwargs["augmentations"] = self.augmentations
        super(MViTacRL_SAC, self).__init__(*args, **kwargs)
        self.beta = beta
        self.tau = tau
        self.lambda_vis = lambda_vis
        self.lambda_tac = lambda_tac
        self.lambda_vis_tac = lambda_vis_tac

    # def _setup_model(self) -> None:
    #     super(RAD_SAC, self)._setup_model()
    #
    #     try:
    #         n_stack = self.env.n_stack
    #     except:
    #         n_stack = 1
    #
    #     self.replay_buffer = MMAugmentedDictReplayBuffer(
    #         self.buffer_size,
    #         self.observation_space,
    #         self.action_space,
    #         self.device,
    #         augmentations=self.augmentations,
    #         augmentations_tactile=self.augmentations_tactile,
    #         visualise_aug=self.visualise_aug,
    #         n_stack=n_stack,
    #         n_envs=self.n_envs,
    #         optimize_memory_usage=self.optimize_memory_usage,
    #     )
    def train(self, gradient_steps: int, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train(gradient_steps, batch_size)

        # Train CURL
        if self.augmentations is not None:
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs = replay_data.observations
            # preprocess the observations
            obs = preprocess_obs(obs, self.observation_space, normalize_images=True)

            # get the tactile observations and convert to dict
            tactile_obs = obs['tactile']
            # get the visual observations
            visual_obs = obs['visual']

            # compute the infoNCE loss
            losses = self.policy.actor.features_extractor.compute_loss(visual_obs, tactile_obs)
            infoNCE_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter = losses

            # weighted sum of the losses using the lambda parameters
            infoNCE_loss = (self.lambda_vis * vis_loss_intra
                            + self.lambda_tac * tac_loss_intra
                            + self.lambda_vis_tac * vis_tac_inter + self.lambda_vis_tac * tac_vis_inter)

            # weight the losses
            vis_loss_intra = vis_loss_intra * self.beta

            # Backpropagate CURL loss and update the online encoder
            self.policy.actor.optimizer.zero_grad()
            infoNCE_loss.backward()
            self.policy.actor.optimizer.step()

            # Soft update of the momentum encoder
            self.actor.features_extractor.momentum_update_key_encoder()

            self.logger.record("train/infoNCE_loss", infoNCE_loss.item())
            self.logger.record("train/vis_loss_intra", vis_loss_intra.item())
            self.logger.record("train/tac_loss_intra", tac_loss_intra.item())
            self.logger.record("train/vis_tac_inter", vis_tac_inter.item())
            self.logger.record("train/tac_vis_inter", tac_vis_inter.item())

class MViTacRL_PPO(RAD_PPO):
    def __init__(
            self,
            *args,
            beta: float = 0.1,  # Coefficient for CURL loss
            tau: float = 0.05,  # Soft update coefficient for CURL
            lambda_vis: float = 1,
            lambda_tac: float = 1,
            lambda_vis_tac: float = 1,
            # augmentations: Dict[str, th.nn.Sequential] = None,
            **kwargs
    ):
        # self.augmentations = augmentations
        # self.augmentations = augmentations["visual"]
        # self.augmentations_tactile = augmentations["tactile"]
        # kwargs["augmentations"] = self.augmentations
        super(MViTacRL_PPO, self).__init__(*args, **kwargs)
        self.beta = beta
        self.tau = tau
        self.lambda_vis = lambda_vis
        self.lambda_tac = lambda_tac
        self.lambda_vis_tac = lambda_vis_tac

    # def _setup_model(self) -> None:
    #     super(RAD_SAC, self)._setup_model()
    #
    #     try:
    #         n_stack = self.env.n_stack
    #     except:
    #         n_stack = 1
    #
    #     self.replay_buffer = MMAugmentedDictReplayBuffer(
    #         self.buffer_size,
    #         self.observation_space,
    #         self.action_space,
    #         self.device,
    #         augmentations=self.augmentations,
    #         augmentations_tactile=self.augmentations_tactile,
    #         visualise_aug=self.visualise_aug,
    #         n_stack=n_stack,
    #         n_envs=self.n_envs,
    #         optimize_memory_usage=self.optimize_memory_usage,
    #     )
    def train(self, batch_size: Optional[int] = None) -> None:
        # Train the actor-critic as usual
        super().train()

        # Train CURL
        if self.augmentations is not None:
            # Sample from the rollout buffer
            # buffer_size = len(self.rollout_buffer.returns)  # Assuming `returns` attribute stores the buffer size
            # batch_inds = np.random.randint(0, buffer_size, size=batch_size)  # Random indices for sampling
            # replay_data = self.rollout_buffer._get_samples(batch_inds)
            for _ in range(self.n_epochs):
                replay_data = self.rollout_buffer.sample(self.batch_size, env=self._vec_normalize_env)
                obs = replay_data.observations
                # preprocess the observations
                obs = preprocess_obs(obs, self.observation_space, normalize_images=True)

                # get the tactile observations and convert to dict
                tactile_obs = obs['tactile']
                # get the visual observations
                visual_obs = obs['visual']

                # compute the infoNCE loss
                losses = self.policy.features_extractor.compute_loss(visual_obs, tactile_obs)
                infoNCE_loss, vis_loss_intra, tac_loss_intra, vis_tac_inter, tac_vis_inter = losses

                # weighted sum of the losses using the lambda parameters
                infoNCE_loss = (self.lambda_vis * vis_loss_intra
                                + self.lambda_tac * tac_loss_intra
                                + self.lambda_vis_tac * vis_tac_inter + self.lambda_vis_tac * tac_vis_inter)

                # weight the losses
                vis_loss_intra = vis_loss_intra * self.beta

                # Backpropagate CURL loss and update the online encoder
                self.policy.optimizer.zero_grad()
                infoNCE_loss.backward()
                self.policy.optimizer.step()

                # Soft update of the momentum encoder
                self.policy.features_extractor.momentum_update_key_encoder()

                self.logger.record("train/infoNCE_loss", infoNCE_loss.item())
                self.logger.record("train/vis_loss_intra", vis_loss_intra.item())
                self.logger.record("train/tac_loss_intra", tac_loss_intra.item())
                self.logger.record("train/vis_tac_inter", vis_tac_inter.item())
                self.logger.record("train/tac_vis_inter", tac_vis_inter.item())