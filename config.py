car = {
    "position" : [300, 100],
    "angle" : 90,
    "num_sensors" : 10,
    "speed" : 0,
    "rotation" : 0
    }

track = {
    "block_width" : 50,
    "block_height" : 50,
    "num_blocks_x" : 10,
    "num_blocks_y" : 10
    }

movement = {
    "acceleration" : 0.4,
    "max_speed" : 10,
    "turn_rate" : 0.2,
    "max_turn_rate" : 3
    }

state = {
    "rest" : 0,
    "decelerate" : 1,
    "accelerate" : 2,
    "accel_left" : 1,
    "accel_right" : 2
    }

reward = {
    "new_tile_reward" : 100,
    "same_tile_reward" : -1,
    "crash_reward" : -10000
    }

