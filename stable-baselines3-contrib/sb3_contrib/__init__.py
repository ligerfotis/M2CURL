import os

# from sb3_contrib.qrdqn import QRDQN
# from sb3_contrib.tqc import TQC
from sb3_contrib.rad import RAD_SAC, RAD_PPO

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
