import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym_car_race.SelfDriveEnv import Car, Track
from gym_car_race.training_utils import TensorboardCallback, constant_schedule, linear_schedule, run_experiment, testing, with_changes
from gym_car_race.config import cfg

model_dir = "models/starting_line"
log_dir = "logs/contest_logs"

os.makedirs(model_dir, exist_ok=True)

# On your marks...

# ===================
# Team configurations
# ===================
# Add your team's configuration in this section and set the correct logging dir
# i.e. "from contest.<my_team_name> import cfg as <my_team_name>"

# Import here:
from contest.example_team import cfg as example_team

# Set log dir here:
example_team["training"]["log_dir"] = log_dir


# ================
# Reward Functions
# ================
# This section is optional
# Define your custom reward function in this section and add it to your config

# Do it like this:
@Car.reward_function
def example_team_reward_func(car):
    """Reward function that always returns 1"""
    return 1

# and then add it to your config like this:
example_team["reward"]["function"] = example_team_reward_func

# Get set...

# ==========
# Race Setup
# ===========
# Follow the template to add an entry for your team inside the list below.

teams = [
#   testing("<my_team_name>", with_changes(<my_team_name>), save_as="<my_team_name>", in_dir=model_dir),
    testing("Team Example", with_changes(example_team), save_as="example_team", in_dir=model_dir),
]

run_experiment(
    # Put the entries here (yes, they are arguments to this function)
    *teams,        
    timesteps=1000, 
    render=True,
    trials=3,
    run_after_training=False,
)

# Go!
# Create a pull request against the contest branch to submit your entry

#  ___
#    _-_-  _/\______\\__
# _-_-__  / ,-. -|-  ,-.`-.
# nyoom - `( o )----( o )-'
#           `-'      `-'