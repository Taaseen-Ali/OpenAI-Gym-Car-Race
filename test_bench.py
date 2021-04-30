import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym_car_race.SelfDriveEnv import Car, Track
from gym_car_race.training_utils import TensorboardCallback, constant_schedule, linear_schedule, run_experiment, testing, with_changes
from gym_car_race.config import cfg

import math


def get_exp(beta):
    return lambda x : (1/beta) * math.exp(-1*x/beta)

def midline_index(obs):
    sensor_data = [entry[3] for entry in obs]    
    exp = get_exp(len(sensor_data)//4) 
    
    total = 0
    for i in range(len(sensor_data)//2):        
        diff = abs(sensor_data[i] - sensor_data[-1*i -1])
        length = sensor_data[i] + sensor_data[-1*i -1]
        percent_diff = diff/length if length else 1
        weighted_diff = exp(i) * percent_diff
        total += weighted_diff
    
    return 1 - total

@Car.reward_function
def reward_func(car):    
    # TODO: 
    # - try increasing min speed and decreasing turn radius
    # - midline index
    # - try rewards depending on action taken instead of state
    # - try including distance from end
    reward = 0
    if car.crashed:
        reward = car.config["reward"]["crash_reward"]
    elif car.has_finished:
        print('finished!')
        reward += 100
    else:
        curr = car.current_tile()
        if curr not in car.traveled:
            reward = car.config["reward"]["new_tile_reward"] * midline_index(car.sensors)
        elif car.speed <= car.config["reward"]["min_speed"]:
            reward = car.config["reward"]["same_tile_penalty"]
        else:
            reward = car.config["reward"]["same_tile_reward"]
    return reward


# Specify folders to save models/logs in

model_dir = "models/tests/alternate-track-new-turning"

os.makedirs(model_dir, exist_ok=True)

# Define tests

run_experiment(

    testing("learning rate .0003",            
        with_changes(
                {
                    "reward": {
                        "function": reward_func,
                    },
                    "car": {
                        "turn_rate": 0.03, 
                        "max_turn_rate": 0.8 
                    },
                    "training": {
                        "learning_rate": 0.0003
                    }
                },
        ),
        save_as="midline-weighted-no-len-lrate-3",        
        in_dir=model_dir),
    
    testing("learning rate .0002",                
        with_changes(
                {
                    "reward": {
                        "function": reward_func,
                    },
                    "car": {
                        "turn_rate": 0.03, 
                        "max_turn_rate": 0.8 
                    },
                    "training": {
                        "learning_rate": 0.0002
                    }
                },
        ),        
        save_as="midline-weighted-no-len-lrate-2",
        in_dir=model_dir),
    

    testing("learning rate .0001",                
        with_changes(
                {
                    "reward": {
                        "function": reward_func,
                    },
                    "car": {
                        "turn_rate": 0.03, 
                        "max_turn_rate": 0.8 
                    },
                    "training": {
                        "learning_rate": 0.0001
                    }
                },
        ),
        save_as="midline-weighted-no-len-lrate-1",
        in_dir=model_dir),
    
    timesteps=100000, 
    render=True,
    trials=10,
    run_after_training=False
)