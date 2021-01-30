"""SelfDrive Environement config

Use this file to help customize various aspects of the simulations. If you find
additional things in the simulation that you'd like to customize (and think
others probably would as well) feel free to add it in and open a pull request on
the repo: https://github.com/Taaseen-Ali/OpenAI-Gym-Car-Race
"""
                                
# Default car settings

car = {
    "position" : [300, 100],    # Starting x, y position of the car. Note that
                                # this will be deprecated soon because of
                                # https://github.com/Taaseen-Ali/OpenAI-Gym-Car-Race/pull/7

    "angle" : 90,               # Initial direction of the car
    "num_sensors" : 10,         # Number of sensor percepts (red dots)
    "speed" : 0,                # Starting speed/angular velocity (these will be
    "rotation" : 0              # removed soon)

    }


# Default track settings

track = {
    "block_width" : 50,         # Width and height of each individual block
    "block_height" : 50,
    "num_blocks_x" : 10,        # Number of blocks in the x/y direction
    "num_blocks_y" : 10
    }

# Car movement settings

movement = {
    "acceleration" : 0.4,       # The accelration of the car
    "max_speed" : 10,           # Top speed 
    "turn_rate" : 0.2,          # How quickly the car is able to turn
    "max_turn_rate" : 3         # Maximum turning ability of the car    
    }

# Mapping of actions to numerical action state values 
# You probably won't have to change these unless the particular method you are
# using requires an action state ranging from particular values

action = {
    "rest" : 0,
    "decelerate" : 1,
    "accelerate" : 2,
    "accel_left" : 1,
    "accel_right" : 2
    }

# Reward values
                                
reward = {
    "new_tile_reward" : 100,    # Reward for reaching a new tile
    "same_tile_reward" : -1,    # Penalty for not moving to a new tile
    "crash_reward" : -10000     # Penalty for crashing
    }

