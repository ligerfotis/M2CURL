# M2CURL: Sample-Efficient Multimodal Reinforcement Learning via Self-Supervised Representation Learning for Robotic Manipulation

This repository contains the code for the paper "[M2CURL: Sample-Efficient Multimodal Reinforcement Learning via Self-Supervised Representation Learning for Robotic Manipulation](https://arxiv.org/abs/2401.17032)"

Authors: [Fotios Lygerakis](https://cps.unileoben.ac.at/fotios-lygerakis-m-sc/), [Vedant Dave](https://cps.unileoben.ac.at/m-sc-vedant-dave/), and [Elmar Rueckert](https://cps.unileoben.ac.at/prof-elmar-rueckert/).

Project page: [https://sites.google.com/view/M2CURL/home](https://sites.google.com/view/M2CURL/home)

The code is based on the [Tactile Gym 2](https://github.com/ac-93/tactile_gym) repository. For details about the Tactile Gym 2 simulator please visit their repository.

## Clone the Repository and Setup Environment

To get started, clone this repository to your local machine and set up a Python virtual environment or a Conda environment:

### Clone the repository
      git clone https://github.com/your_username/m2curl_project.git
      cd m2curl_project

### Create and activate a Python virtual environment
      python -m venv env
      source env/bin/activate

### Or create and activate a Conda environment
      conda create --name m2curl_env python=3.10
      conda activate m2curl_env

## Installation

### Prerequisites

Make sure you have the following prerequisites installed on your system:

- Python 3.10
- Ubuntu 22.04
- CUDA (>=11.8) if you're using GPU acceleration

### Installation Steps

1. Install PyTorch and its dependencies:

    ```
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    ```

2. Set the CUDA library path:

    ```
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ```

3. Install setuptools and wheel:

    ```
    pip install setuptools==65.5.0 "wheel<0.40.0"
    ```

4. Install the project:

    ```
    python setup.py install
    ```

5. Navigate to the stable-baselines3-contrib directory:

    ```
    cd stable-baselines3-contrib/
    ```

6. Install stable-baselines3-contrib:

    ```
    python setup.py install
    ```

## Training

To train an RL agent, follow these steps:

1. Navigate to the sb3_helpers directory:

    ```
    cd ../tactile_gym/sb3_helpers/
    ```

2. Run the training script:

    ```
    python train_agent.py -A mvitac_sac -E object_push-v0 -O visuotactile --total_timesteps 500_000
    ```

Replace the arguments `-A`, `-E`, `-O`, and `--total_timesteps` with your desired values. Here's the list of available arguments:

    - `-A`, `--algo_name`: Algorithm name (`ppo`, `sac`, `rad_ppo`, `rad_sac`, `mvitac_sac`, `mvitac_ppo`).
    - `-E`, `--env_name`: Environment name.
    - `-O`, `--observation_mode`: Observation mode.
    - `-S`, `--tactile_sensor_name`: Tactile sensor name.
    - `--total_timesteps`: Total training timesteps.
    - `--n_stack`: Number of frames to stack.
    - `--use_wandb`: Whether to use Wandb for logging (True/False).
    - `--eval_freq`: Evaluation frequency.
    - `--buffer_size`: Replay buffer size.
    - `--n_envs`: Number of parallel environments.
    - `--learning_starts`: Learning starts.
    - `--n_steps`: Number of steps per gradient update.
    - `--beta`: Beta value (for `mvitac_sac` and `mvitac_ppo` algorithms).
    - `--reward_mode`: Reward mode.
    - `--seed`: Random seed.
    - `--lambda_visual`: Lambda value for visual features (for `mvitac_sac` and `mvitac_ppo` algorithms).
    - `--lambda_tactile`: Lambda value for tactile features (for `mvitac_sac` and `mvitac_ppo` algorithms).
    - `--lambda_visual_to_tactile`: Lambda value for visual-to-tactile features (for `mvitac_sac` and `mvitac_ppo` algorithms).
    - `--tau`: Temperature (for `mvitac_sac` and `mvitac_ppo` algorithms).

Cite this work as:

```
@INPROCEEDINGS{10597462,
  author={Lygerakis, Fotios and Dave, Vedant and Rueckert, Elmar},
  booktitle={2024 21st International Conference on Ubiquitous Robots (UR)}, 
  title={M2CURL: Sample-Efficient Multimodal Reinforcement Learning via Self-Supervised Representation Learning for Robotic Manipulation}, 
  year={2024},
  volume={},
  number={},
  pages={490-497},
  keywords={Representation learning;Visualization;Heuristic algorithms;Scalability;Reinforcement learning;Self-supervised learning;Robot sensing systems},
  doi={10.1109/UR61395.2024.10597462}}
```
