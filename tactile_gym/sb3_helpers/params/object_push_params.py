import torch.nn as nn
import kornia.augmentation as K
from stable_baselines3.common.torch_layers import NatureCNN
from torchvision.transforms import transforms

from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor
from tactile_gym.sb3_helpers.encoder_arsenal import MViTacFeatureExtractor, MViTacFeatureExtractor2

# ============================== RAD ==============================
augmentations = nn.Sequential(
    K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),
)

# # Define the random crop and normalization transforms
# augmentations = transforms.Compose([
#     transforms.RandomCrop(size=(84, 84)),
#     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Example normalization values
# ])
#
augmentations_tactile = transforms.Compose([
    transforms.RandomCrop(size=(84, 84)),
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Example normalization values
])

# ============================== PPO ==============================
rl_params_ppo = {
    # ==== env params ====
    "algo_name": "ppo",
    "env_name": "object_push-v0",
    "max_ep_len": 350,
    "image_size": [128, 128],
    "env_modes": {
        ## which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        # 'movement_mode':'xyRz',
        "movement_mode": "TyRz",
        # 'movement_mode':'TxTyRz',

        # specify the arm and the tactile sensor
        # "arm_type": 'mg400',
        "arm_type": 'ur5',
        "tactile_sensor_name": 'digit',
        # "tactile_sensor_name": 'tactip',
        # "tactile_sensor_name": 'digitac',

        ## the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        ## randomisations
        "rand_init_orn": False,
        'rand_obj_mass': False,

        ## straight or random trajectory
        # 'traj_type':'straight',
        "traj_type": "simplex",

        ## which observation type to return
        # 'observation_mode': 'oracle',
        "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        ## the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    },
    # ==== control params ====
    'policy': 'MultiInputPolicy',
    "seed": int(1),
    "n_stack": 3,
    "total_timesteps": int(1e5),
    "n_eval_episodes": 10,
    "n_envs": 10,
    "eval_freq": 1e4,
}

ppo_params = {
    # === net arch ===
    "policy_kwargs": {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            'cnn_base': NatureCNN,
            'cnn_output_dim': 256,
            'mlp_extractor_net_arch': [64, 64],
        },
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn": nn.Tanh,
    },

    # ==== rl params ====
    "learning_rate": 3e-4,
    "n_steps": int(2048),
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": 0.1,
}
# ============================== PPO ==============================
rl_params_mvitacrl_ppo = {
    # ==== env params ====
    "algo_name": "MViTacRL_ppo",
    "env_name": "object_push-v0",
    "max_ep_len": 350,
    "image_size": [128, 128],
    "env_modes": {
        ## which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        # 'movement_mode':'xyRz',
        "movement_mode": "TyRz",
        # 'movement_mode':'TxTyRz',

        # specify the arm and the tactile sensor
        # "arm_type": 'mg400',
        "arm_type": 'ur5',
        "tactile_sensor_name": 'digit',
        # "tactile_sensor_name": 'tactip',
        # "tactile_sensor_name": 'digitac',

        ## the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        ## randomisations
        "rand_init_orn": False,
        'rand_obj_mass': False,

        ## straight or random trajectory
        # 'traj_type':'straight',
        "traj_type": "simplex",

        ## which observation type to return
        # 'observation_mode': 'oracle',
        "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        # 'observation_mode':'visuotactile_and_feature',

        ## the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    },
    # ==== control params ====
    'policy': 'MultiInputPolicy',
    "seed": int(1),
    "n_stack": 3,
    "total_timesteps": int(1e5),
    "n_eval_episodes": 10,
    "n_envs": 2,
    "eval_freq": 1e4,
}

mvitac_rl_ppo_params = {
    # === net arch ===
    "policy_kwargs": {
        "features_extractor_class": MViTacFeatureExtractor2,
        "features_extractor_kwargs": {
            'cnn_output_dim': 512,
            'mlp_extractor_net_arch': [64, 64],
            'mm_hyperparams': dict(
                n_channels_vision=3,
                n_channels_touch=1,
                intra_dim=128,
                inter_dim=128,
                temperature=0.1,
                weight_intra_vision=1,
                weight_intra_tactile=1,
                weight_inter_tac_vis=1,
                weight_inter_vis_tac=1)
        },
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn": nn.Tanh,
    },

    # ==== rl params ====
    "learning_rate": 3e-4,
    "n_steps": int(2048),
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.95,
    "gae_lambda": 0.9,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": 0.1,
}

# ============================== SAC ==============================

rl_params_sac = {
    # ==== env params ====
    "algo_name": "sac",
    "env_name": "object_push-v0",
    "max_ep_len": 350,
    "image_size": [128, 128],
    "env_modes": {
        ## which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        # 'movement_mode':'xyRz',
        "movement_mode": "TyRz",
        # 'movement_mode':'TxTyRz',
        "arm_type": 'ur5',
        "tactile_sensor_name": 'digit',
        ## the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        ## randomisations
        "rand_init_orn": False,
        'rand_obj_mass': False,

        ## straight or random trajectory
        # 'traj_type':'straight',
        "traj_type": "simplex",

        ## which observation type to return
        # 'observation_mode': 'oracle',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        'observation_mode': 'visuotactile',

        ## the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    },
    # ==== control params ====
    'policy': 'MultiInputPolicy',
    "n_stack": 1,
    "seed": int(1),
    "total_timesteps": int(1e6),
    "n_eval_episodes": 10,
    "n_envs": 10,
    "eval_freq": 1e4,
}

sac_params = {
    # === net arch ===
    "policy_kwargs": {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {
            'cnn_output_dim': 256,
            'mlp_extractor_net_arch': [64, 64],
        },
        "net_arch": dict(pi=[256, 256], qf=[256, 256]),
        "activation_fn": nn.Tanh,
    },

    # ==== rl params ====
    "learning_rate": 3e-4,
    "buffer_size": int(2e4),
    "learning_starts": 1e4,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
    "optimize_memory_usage": False,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
}
rl_params_mvitacrl_sac = {
    # ==== env params ====
    "algo_name": "MViTacRL_sac",
    "env_name": "object_push-v0",
    "max_ep_len": 350,
    "image_size": [128, 128],
    "env_modes": {
        ## which dofs can have movement (environment dependent)
        # 'movement_mode':'y',
        # 'movement_mode':'yRz',
        # 'movement_mode':'xyRz',
        "movement_mode": "TyRz",
        # 'movement_mode':'TxTyRz',
        "arm_type": 'ur5',
        "tactile_sensor_name": 'digit',
        ## the type of control used
        # 'control_mode':'TCP_position_control',
        "control_mode": "TCP_velocity_control",

        ## randomisations
        "rand_init_orn": False,
        'rand_obj_mass': False,

        ## straight or random trajectory
        # 'traj_type':'straight',
        "traj_type": "simplex",

        ## which observation type to return
        # 'observation_mode': 'oracle',
        # "observation_mode": "tactile_and_feature",
        # 'observation_mode':'visual_and_feature',
        'observation_mode': 'visuotactile',

        ## the reward type
        "reward_mode": "dense"
        # 'reward_mode':'sparse'
    },
    # ==== control params ====
    'policy': 'MultiInputPolicy',
    "n_stack": 3,
    "seed": int(1),
    "total_timesteps": int(1e6),
    "n_eval_episodes": 10,
    "n_envs": 10,
    "eval_freq": 1e4,
}

mvitac_rl_sac_params = {
    # === net arch ===
    "policy_kwargs": {
        "features_extractor_class": MViTacFeatureExtractor2,
        "features_extractor_kwargs": {
            'cnn_output_dim': 512,
            'mlp_extractor_net_arch': [64, 64],
            'mm_hyperparams': dict(
                n_channels_vision=3,
                n_channels_touch=1,
                intra_dim=128,
                inter_dim=128,
                temperature=0.1,
                weight_intra_vision=1,
                weight_intra_tactile=1,
                weight_inter_tac_vis=1,
                weight_inter_vis_tac=1)
        },
        "net_arch": dict(pi=[256, 256], qf=[256, 256]),
        "activation_fn": nn.Tanh,
    },

    # ==== rl params ====
    "learning_rate": 3e-4,
    "buffer_size": int(2e4),
    "learning_starts": 1e2,
    "batch_size": 64,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "action_noise": None,
    "optimize_memory_usage": False,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
}
