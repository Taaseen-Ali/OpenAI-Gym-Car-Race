import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from SelfDriveEnv import Car, Track
from training_utils import TensorboardCallback, constant_schedule, linear_schedule, run_experiment, testing, with_changes
from config import cfg


@Car.reward_function
def reward_func(car):
    reward = 0
    if car.crashed:
        reward = car.config["reward"]["crash_reward"]
    elif car.has_finished:
        print('finished!')
        reward += 100
    else:
        curr = car.current_tile()
        if curr not in car.traveled:
            reward = car.config["reward"]["new_tile_reward"] * (len(car.traveled) + 1)
        elif car.speed <= car.config["reward"]["min_speed"]:
            reward = car.config["reward"]["same_tile_penalty"]
        else:
            reward = car.config["reward"]["same_tile_reward"]
    return reward


# Specify folders to save models/logs in

model_dir = "02_same-tile-penalty"
log_dir = "test-bench-logs"

os.makedirs(model_dir, exist_ok=True)

# Define tests

run_experiment(

    testing("same tile penalty as -8",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -8
                    }                    
                }
            ),
        save_as="same-tile-80",
        in_dir=model_dir),

    testing("same tile penalty as -4",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -4
                    }                    
                }
            ),
        save_as="same-tile-40",
        in_dir=model_dir),

    testing("same tile penalty as -1",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -1
                    }                    
                }
            ),
        save_as="same-tile-10",
        in_dir=model_dir),

    testing("same tile penalty as -.5",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -.5
                    }                    
                }
            ),
        save_as="same-tile-5",
        in_dir=model_dir),

    testing("same tile penalty as -.2",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -.2
                    }                    
                }
            ),
        save_as="same-tile-2",
        in_dir=model_dir),

    testing("same tile penalty as -.1",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,                        
                    }                    
                }
            ),
        save_as="same-tile-1",
        in_dir=model_dir),
        
    testing("same tile penalty as -.07",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -.07
                    }                    
                }
            ),
        save_as="same-tile-07",
        in_dir=model_dir),

    testing("same tile penalty as -.05",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },                    
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -.05
                    }          
                }
            ),
        save_as="same-tile-05",
        in_dir=model_dir),

    testing("same tile penalty as -.01",            
        with_changes(
                {
                    "training": {
                        "log_dir": log_dir,
                    },                    
                    "reward": {
                        "function": reward_func,
                        "same_tile_penalty": -.01
                    }          
                }
            ),
        save_as="same-tile-01",
        in_dir=model_dir),


    timesteps=50000, 
    render=True,
    trials=10,
    run_after_training=False,
)
