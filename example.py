import os

import numpy as np
import matplotlib.pyplot as plt

from SelfDriveEnv import Car, Track
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import stable_baselines3 as sb
print(sb.common.logger.Video)

import config as cfg


# Set the directory where your models should be stored as well as the name of
# the model that you want to load/save
# current best is model18, PPO_61
model_dir = "./models/"
model_name = "model18"
os.makedirs(model_dir, exist_ok=True)


# Helper for adjusting the learning rate

def linear_schedule(initial_value: float): 
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value         
    return func


# Helper for logging 

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.count = 0
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        
        value = sum(self.training_env.get_attr("cars")[0][0].reward_history)
        self.logger.record('reward', value)
        return True


# Create custom reward function

@Car.reward_function
def reward_func(car):
    reward = 0
    if car.crashed:
        reward = cfg.reward["crash_reward"]
    elif car.has_finished:
        print('finished!')
        reward += 100
    else:
        curr = car.current_tile()
        if curr not in car.traveled:
            reward = cfg.reward["new_tile_reward"] * (len(car.traveled) + 1)
        elif car.speed <= cfg.car["acceleration"]:
            reward = -3 * cfg.reward["same_tile_reward"]
        else:
            reward = cfg.reward["same_tile_reward"]
    return reward


# Set up the environment using the values found in configs

env = Track(cfg.track['num_blocks_x'], cfg.track['num_blocks_y'], 
            cfg.track['block_width'], cfg.track['block_height'])
car = Car(cfg.car['position'][0], cfg.car['position'][1], 
          cfg.car['num_sensors'], reward_func=reward_func)
env.add_car(car)

# Uncomment this if you've made any changes to the environment and want to make
# sure that everything is still okay (no output means everything is fine):

# check_env(env)

# Set the the total number of steps to train for

timesteps = 50000

# Uncomment one of the following depending on what you'd like to do

# A. Use an existing model
model = PPO.load(model_dir + model_name)

# B. Create and train a new model
# model = PPO('MlpPolicy', env, tensorboard_log="./ppo/", verbose=1, learning_rate=linear_schedule(.005)) #set it to around .001 next time
# model.learn(total_timesteps=timesteps, callback=TensorboardCallback()) 
# model.save(model_dir + model_name)

# C. Load an existing model and keep training with it
# model = PPO.load(model_dir + model_name)
# model.learn(total_timesteps=10000) 
# model.save(model_dir + model_name)

# Reset the env

env = Track(cfg.track['num_blocks_x'], cfg.track['num_blocks_y'], 
            cfg.track['block_width'], cfg.track['block_height'])
car = Car(cfg.car['position'][0], cfg.car['position'][1], 
          cfg.car['num_sensors'], reward_func=reward_func)
env.add_car(car)

obs = env.reset(new=False) # You can set new=True if you'd like to create a new track

# Run the simulation until the car crashes or finishes

done = False
while not done:
    action, _states = model.predict(obs) 
    obs, rewards, done, info = env.step(action)
    env.render()
