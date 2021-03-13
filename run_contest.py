import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym_car_race.SelfDriveEnv import Car, Track
from gym_car_race.training_utils import TensorboardCallback, constant_schedule, linear_schedule, run_experiment, testing, with_changes
from gym_car_race.config import cfg

# On your marks...

# Add your team's configuration in this section.
# ===================
# Team configurations
# ===================

# i.e. "import contest.<my_team_name>.py"


model_dir = "/starting_line/"
log_dir = "/contest_data"

os.makedirs(model_dir, exist_ok=True)


# Define your custom reward function in this section to reference them in the
# training section below (this is optional)

# ================
# Reward Functions
# ================

# Do it like this:
@Car.reward_func
def example_func(car):
    """Reward function that always returns 1"""
    return 1


# Get set...

# ==========
# Race Setup
# ===========
# Add an entry for your team below. Follow this template:

# testing(
#     "<my_team_name>",            
#     with_changes(<my_team_name>.py),
#     save_as="<my_team_name>",
#     in_dir=model_dir
#     ),
# 

run_experiment(
    # Put the entries here (yes, they are arguments to this function)
    
    timesteps=50000, 
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