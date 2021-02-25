"""SelfDrive Environement config
Use this file to help customize various aspects of the simulations. If you find
additional things in the simulation that you'd like to customize (and think
others probably would as well) feel free to add it in and open a pull request on
the repo: https://github.com/Taaseen-Ali/OpenAI-Gym-Car-Race
"""
                                
# Default car settings
cfg = {
    "car": {
        # Initial defaults

        "position": [100, 100],     # Starting x, y position of the car. Note that
                                    # this will be deprecated soon because of
                                    # https://github.com/Taaseen-Ali/OpenAI-Gym-Car-Race/pull/7
        "width": 50,                # Box width/height of the car
        "height": 50,               
        "angle": 90,                # Initial direction of the car
        "num_sensors": 11,          # Number of sensor percepts (dots radiating out from the car)
        "sensor_color": (255, 255, 0),
        "speed": 0,                 # Starting speed/angular velocity (these will be
        "rotation": 0,              # removed soon)
        "image": "gym_car_race/images/cars/Audi.png",    # Path to image file for rendering the car
        
        # Car movement
        
        "acceleration": 0.4,        # The acceleration of the car
        "max_speed": 5,             # Top speed 
        "turn_rate": 0.2,           # How quickly the car is able to turn
        "max_turn_rate": 4          # Maximum turning ability of the car 
        },


    # Default track settings

    "track": {
        "block_width": 25,          # Width and height of each individual block
        "block_height": 25,
        "num_blocks_x": 20,         # Number of blocks in the x/y direction
        "num_blocks_y": 20, 
        "start_line_color": (0, 128, 0),            # Staring/finish/normal block colors
        "finish_line_color": (255, 0, 0),
        "default_color": (87, 46, 140),
        "border_color": (255, 255, 255),            # Color outlining each blocks
        "track_file": "gym_car_race/track.csv",     # Path to track file
        },

    # Mapping of actions to numerical action state values 
    # You probably won't have to change these unless the particular method you are
    # using requires an action state ranging from particular values

    "action": {
        "rest" : 0,
        "decelerate" : 1,
        "accelerate" : 2,
        "accel_left" : 1,
        "accel_right" : 2
        },

    # Reward values
                                    
    "reward": {
        "new_tile_reward" : 10,     # Reward for reaching a new tile
        "min_speed": 0,           
        "same_tile_reward" : .1,    # Reward for staying alive
        "same_tile_penalty": -.1,   # Penalty for not moving to a new tile
        "crash_reward" : -100,       # Penalty for crashing
        "finish_reward": 100,       # Reward for reaching finish line
        
        "function": None,           # Reward function to use. Default 
                                    # implementation will be used if 
                                    # none is specified 
        },

    # Training default configs
    
    "training": {
        "learning_rate": lambda progress: .0003,    # Schedule function for specifying 
                                                    # learning rate. Progress is a float 
                                                    # from 0-1 denoting how much of the 
                                                    # training has been completed relative 
                                                    # to total_timesteps
        
        "log_dir": "logs"                           # Directory to store tensorboard logs
        }
}
