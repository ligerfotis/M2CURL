# import gym
import os
import sys
import time
# import numpy as np

# import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps, CheckpointCallback

from stable_baselines3 import PPO, SAC
from sb3_contrib import RAD_SAC, RAD_PPO

import tactile_gym.rl_envs
from tactile_gym.sb3_helpers.custom_policies import MViTacRL_SAC, MViTacRL_PPO
from tactile_gym.sb3_helpers.params import import_parameters
from tactile_gym.sb3_helpers.rl_utils import make_training_envs, make_eval_env
from tactile_gym.sb3_helpers.eval_agent_utils import final_evaluation
from tactile_gym.utils.general_utils import (
    save_json_obj,
    load_json_obj,
    convert_json,
    check_dir,
)
from tactile_gym.sb3_helpers.custom.custom_callbacks import (
    FullPlottingCallback,
    ProgressBarManager,
)
import argparse

import torch.nn as nn
import kornia.augmentation as K
from stable_baselines3.common.torch_layers import NatureCNN
from tactile_gym.sb3_helpers.custom.custom_torch_layers import CustomCombinedExtractor
import wandb
from wandb.integration.sb3 import WandbCallback

# ============================== RAD ==============================
augmentations = nn.Sequential(
    K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),
)


parser = argparse.ArgumentParser(description="Train an agent in a tactile gym task.")
# metavar ='' can tidy the help tips.
# parser.add_argument("-E", '--env_name', type=str, required = True, help='The name of a tactile gym env.', metavar='')
parser.add_argument("-M", '--movement_mode', type=str, help='The movement mode.', metavar='')
parser.add_argument("-T", '--traj_type', type=str, help='The traj type.', metavar='')
parser.add_argument("-R", '--retrain_path', type=str, help='Retrain model path.', metavar='')
parser.add_argument("-I", '--if_retrain', type=str, help='Retrain.', metavar='')
# algorithm name
parser.add_argument("-A", '--algo_name', type=str, help='Algorithm name.', metavar='', default=None)
# env name
parser.add_argument("-E", '--env_name', type=str, help='Env name.', metavar='', default=None)
# observation mode
parser.add_argument("-O", '--observation_mode', type=str, help='Observation mode.', metavar='', default=None)
# tactic sensor name
parser.add_argument("-S", '--tactile_sensor_name', type=str, help='Tactile sensor name.', metavar='', default=None)
# timesteps
parser.add_argument('--total_timesteps', type=int, help='Total timesteps.', metavar='', default=None)
# number of frames to stack
parser.add_argument('--n_stack', type=int, help='Number of frames to stack.', metavar='', default=None)
# use wandb
parser.add_argument('--use_wandb', type=bool, help='Use wandb.', metavar='', default=False)
# eval freq
parser.add_argument('--eval_freq', type=int, help='Eval freq.', metavar='', default=None)
# buffer size
parser.add_argument('--buffer_size', type=int, help='Buffer size.', metavar='', default=None)
# number of envs
parser.add_argument('--n_envs', type=int, help='Number of envs.', metavar='', default=None)
# learning starts
parser.add_argument('--learning_starts', type=int, help='Learning starts.', metavar='', default=None)
# num of steps
parser.add_argument('--n_steps', type=int, help='Num of steps.', metavar='', default=None)
# beta
parser.add_argument('--beta', type=float, help='Beta.', metavar='', default=None)
# reward mode
parser.add_argument('--reward_mode', type=str, help='Reward mode.', metavar='', default=None)
# seed
parser.add_argument('--seed', type=int, help='Seed.', metavar='', default=None)
# lambda visual
parser.add_argument('--lambda_visual', type=float, help='Lambda visual.', metavar='', default=1)
# lambda tactile
parser.add_argument('--lambda_tactile', type=float, help='Lambda tactile.', metavar='', default=1)
# lambda visual to tactile
parser.add_argument('--lambda_visual_to_tactile', type=float, help='Lambda visual to tactile.', metavar='', default=1)
# temperature
parser.add_argument('--tau', type=float, help='Temperature.', metavar='', default=1)

args = parser.parse_args()


def fix_floats(data):
    if isinstance(data, list):
        iterator = enumerate(data)
    elif isinstance(data, dict):
        iterator = data.items()
    else:
        raise TypeError("can only traverse list or dict")

    for i, value in iterator:
        if isinstance(value, (list, dict)):
            fix_floats(value)
        elif isinstance(value, str):
            try:
                data[i] = float(value)
            except ValueError:
                pass


def train_agent(
    algo_name="ppo",
    env_name="edge_follow-v0",
    rl_params={},
    algo_params={},
    augmentations=None,
):
    # check if running in debug mode
    if sys.gettrace() is not None:
        # set use_wandb to False
        args.use_wandb = False
        # set number of envs to 2
        rl_params["n_envs"] = 2
        # rl_params["eval_freq"] = 1000

    # create save dir
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # change directory to the project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    save_dir = os.path.join(
        "saved_models/", rl_params["env_name"], timestr, algo_name, "s{}_{}".format(
            rl_params["seed"], rl_params["env_modes"]["observation_mode"])
    )


    check_dir(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # save params
    save_json_obj(convert_json(rl_params), os.path.join(save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(save_dir, "algo_params"))
    if "rad" in algo_name or "MViTacRL" in algo_name:
        save_json_obj(convert_json(augmentations), os.path.join(save_dir, "augmentations"))

    # config for wandb
    config = {
        "env_name": env_name,
        "algo_name": algo_name,
        "rl_params": rl_params,
        "algo_params": algo_params,
        "augmentations": augmentations,
    }

    # init wandb
    if args.use_wandb:
        run = wandb.init(
            project="mvitacrl",
            config=config, 
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
    else:
        # create a dummy run object
        run = type("run", (object,), {"id": "debug"})

    # load the envs
    env = make_training_envs(env_name, rl_params, save_dir)

    eval_env = make_eval_env(
        env_name,
        rl_params,
        show_gui=False,
        show_tactile=False,
    )

    # define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "trained_models/"),
        log_path=os.path.join(save_dir, "trained_models/"),
        eval_freq=rl_params["eval_freq"],
        n_eval_episodes=rl_params["n_eval_episodes"],
        deterministic=True,
        render=True,
        verbose=1,
    )

    # callback for wandb
    wandb_callback = None
    if args.use_wandb:
        wandb_callback = WandbCallback(
            gradient_save_freq=rl_params["eval_freq"] * rl_params["n_envs"],
            model_save_path=f"models/{run.id}",
            verbose=2,
            log="gradients",
        )

    plotting_callback = FullPlottingCallback(log_dir=save_dir, total_timesteps=rl_params["total_timesteps"])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params["eval_freq"] * rl_params["n_envs"], callback=plotting_callback)

    # create the model with hyper params
    if algo_name == "ppo":
        model = PPO(rl_params["policy"], env, **algo_params, verbose=1, tensorboard_log=f"runs/{run.id})")
    elif algo_name == "rad_ppo":
        model = RAD_PPO(rl_params["policy"], env, **algo_params, augmentations=augmentations, visualise_aug=False, verbose=1, tensorboard_log=f"runs/{run.id})")
    elif algo_name == "sac":
        model = SAC(rl_params["policy"], env, **algo_params, verbose=1, tensorboard_log=f"runs/{run.id})")
    elif algo_name == "rad_sac":
        model = RAD_SAC(rl_params["policy"], env, **algo_params, augmentations=augmentations, visualise_aug=False, verbose=1, tensorboard_log=f"runs/{run.id})")
    elif algo_name == "mvitac_sac":
        model = MViTacRL_SAC(rl_params["policy"], env, **algo_params, augmentations=augmentations,
                             visualise_aug=False, verbose=1, tensorboard_log=f"runs/{run.id})")
        # model = MViTacRL_SAC("CnnPolicy", env, policy_kwargs=, verbose=1, buffer_size=buffer_size,
        #              train_freq=1,
        #              gradient_steps=-1, augmentations=augmentations, visualise_aug=False,
        #              tensorboard_log=f"runs/{run.id}", batch_size=hyperparam_dict["batch_size"])
    elif algo_name == "mvitac_ppo":
        model = MViTacRL_PPO(rl_params["policy"], env, **algo_params, augmentations=augmentations,
                             visualise_aug=False, verbose=1, tensorboard_log=f"runs/{run.id})")
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))

    # train an agent
    with ProgressBarManager(rl_params["total_timesteps"]) as progress_bar_callback:
        if args.use_wandb:
            callback_list = [progress_bar_callback, eval_callback, event_plotting_callback, wandb_callback]
        else:
            callback_list = [progress_bar_callback, eval_callback, event_plotting_callback]
        model.learn(
            total_timesteps=rl_params["total_timesteps"],
            callback=callback_list,
        )

    # save the final model after training
    model.save(os.path.join(save_dir, "trained_models", "final_model"))
    env.close()
    eval_env.close()

    # run final evaluation over 20 episodes and save a vid
    final_evaluation(
        saved_model_dir=save_dir,
        n_eval_episodes=10,
        seed=None,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        render=True,
        save_vid=True,
        take_snapshot=False,
    )


def retrain_agent(model_path,
                  algo_name='ppo',
                  env_name='edge_follow-v0',
                  rl_params={},
                  algo_params={},
                  augmentations=None,):

    timestr = time.strftime("%Y%m%d-%H%M%S")

    new_save_dir = os.path.join(
        "saved_models/", "retrain_models/", env_name, timestr, algo_name, "s{}_{}".format(
            rl_params["seed"], rl_params["env_modes"]["observation_mode"])
    )
    check_dir(new_save_dir)
    os.makedirs(new_save_dir, exist_ok=True)
    # save params
    save_json_obj(convert_json(rl_params), os.path.join(new_save_dir, "rl_params"))
    save_json_obj(convert_json(algo_params), os.path.join(new_save_dir, "algo_params"))
    if 'rad' in algo_name:
        save_json_obj(convert_json(augmentations), os.path.join(new_save_dir, "augmentations"))
    # load the envs
    env = make_training_envs(
        env_name,
        rl_params,
        new_save_dir
    )

    eval_env = make_eval_env(
        env_name,
        rl_params,
        show_gui=False,
        show_tactile=False,
    )

    # define callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(new_save_dir, "trained_models/"),
        log_path=os.path.join(new_save_dir, "trained_models/"),
        eval_freq=rl_params["eval_freq"],
        n_eval_episodes=rl_params["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1,
    )

    plotting_callback = FullPlottingCallback(log_dir=new_save_dir, total_timesteps=rl_params['total_timesteps'])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params['eval_freq']*rl_params['n_envs'], callback=plotting_callback)

    save_frequency = 400000
    checkpoint_callback = CheckpointCallback(save_freq=save_frequency / rl_params["n_envs"], save_path=os.path.join(new_save_dir, "trained_models/"),
                                             name_prefix='rl_model')
    # creat agent and load the policy zip file
    if algo_name == 'rad_ppo':
        model = PPO(

            rl_params["env_name"],
            env,
            **algo_params,
            verbose=1
        )
    elif algo_name == 'rad_ppo':
        model = RAD_PPO(
            rl_params["policy"],
            env,
            **algo_params,
            augmentations=augmentations,
            visualise_aug=False,
            verbose=1
        )

        model = model.load(model_path, env=env)
    else:
        sys.exit("Incorrect algorithm specified: {}.".format(algo_name))
    # set_trace()
    # train an agent
    with ProgressBarManager(
        rl_params["total_timesteps"]
    ) as progress_bar_callback:
        model.learn(
            total_timesteps=rl_params["total_timesteps"],
            callback=[progress_bar_callback, eval_callback, event_plotting_callback, checkpoint_callback],
        )

    # save the final model after training
    model.save(os.path.join(new_save_dir, "trained_models", "final_model"))
    env.close()
    eval_env.close()

    # run final evaluation over 20 episodes and save a vid
    final_evaluation(
        saved_model_dir=new_save_dir,
        n_eval_episodes=10,
        seed=None,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        render=True,
        save_vid=True,
        take_snapshot=False
    )


if __name__ == "__main__":

    if args.if_retrain:
        saved_model_dir = args.retrain_path
        model_path = os.path.join(saved_model_dir, "trained_models", "best_model.zip")
        rl_params = load_json_obj(os.path.join(saved_model_dir, "rl_params"))
        algo_params = load_json_obj(os.path.join(saved_model_dir, "algo_params"))

        env_name = rl_params["env_name"]
        algo_name = rl_params["algo_name"]
        # need to load the class
        if algo_params['policy_kwargs']['features_extractor_class'] == "CustomCombinedExtractor":
            algo_params['policy_kwargs']['features_extractor_class'] = CustomCombinedExtractor
        if algo_params['policy_kwargs']['features_extractor_kwargs']['cnn_base'] == "NatureCNN":
            algo_params['policy_kwargs']['features_extractor_kwargs']['cnn_base'] = NatureCNN
        if algo_params['policy_kwargs']['activation_fn'] == "Tanh":
            algo_params['policy_kwargs']['activation_fn'] = nn.Tanh
        if os.path.isfile(os.path.join(saved_model_dir, 'augmentations.json')):
            algo_name = "rad_" + algo_name
            augmentations = nn.Sequential(K.RandomAffine(degrees=0, translate=[0.05, 0.05], scale=[1.0, 1.0], p=0.5),)

        retrain_agent(
            model_path,
            algo_name,
            env_name,
            rl_params,
            algo_params,
            augmentations
        )
    else:
        # if args.algo_name is not None:
        algo_name = args.algo_name
        env_name = args.env_name

        rl_params, algo_params, augmentations = import_parameters(env_name, algo_name)

        if args.observation_mode is not None:
            rl_params['env_modes']['observation_mode'] = args.observation_mode
        if args.tactile_sensor_name is not None:
            rl_params['env_modes']['tactile_sensor_name'] = args.tactile_sensor_name
        if args.total_timesteps is not None:
            rl_params['total_timesteps'] = args.total_timesteps
        if args.n_stack is not None:
            rl_params['n_stack'] = args.n_stack
        if args.eval_freq is not None:
            rl_params['eval_freq'] = args.eval_freq
        if args.buffer_size is not None:
            algo_params['buffer_size'] = args.buffer_size
        if args.n_envs is not None:
            rl_params['n_envs'] = args.n_envs
        if args.learning_starts is not None:
            algo_params['learning_starts'] = args.learning_starts
        if args.n_steps is not None:
            algo_params['n_steps'] = args.n_steps
        if args.beta is not None and algo_name in ['mvitac_sac', 'mvitac_ppo']:
            algo_params['beta'] = args.beta
        if args.reward_mode is not None:
            rl_params['env_modes']['reward_mode'] = args.reward_mode
        if args.reward_mode == 'sparse':
            algo_params['learning_starts'] = 0
        if args.seed is not None:
            rl_params['seed'] = args.seed
        if algo_name in ['mvitac_sac', 'mvitac_ppo']:
            algo_params['lambda_vis'] = args.lambda_visual
            algo_params['lambda_tac'] = args.lambda_tactile
            algo_params['lambda_vis_tac'] = args.lambda_visual_to_tactile
            algo_params['policy_kwargs']['features_extractor_kwargs']['mm_hyperparams']['temperature'] = args.tau


        # import paramters
        train_agent(
            algo_name,
            env_name,
            rl_params,
            algo_params,
            augmentations
        )
