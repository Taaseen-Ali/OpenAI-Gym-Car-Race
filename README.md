<img src="./gym_car_race/images/logo.svg"  align="right" width="30%"/>

# OpenAI Gym Car Race

An OpenAI Gym environment for simulating a self driving car on a race track. Built for [NYU SelfDrive](https://engineering.nyu.edu/research/vertically-integrated-projects/vip-teams/nyu-self-drive).

This project is still under active development. [Bug reports](https://github.com/Taaseen-Ali/OpenAI-Gym-Car-Race/issues) are welcome.

## Prerequisites

Ensure that you are using python3 and that all required python modules are installed by running the following:

`pip install --no-cache-dir -r requirements.txt`

## Running

Run `python example.py` in the root of this repository to execute the example project. Doing so will create the necessary folders and begin the process of training a simple nueral network. After training has completed, a window will open showing the car navigating the pre-saved track using the trained model.

To create a new track, pass in `new=True` to the reset function of the environment. Doing so will open a window inside of which you have the following options:

- Left click: remove a block
- Right click: put back a block
- `S` key: put down a green start block
- `F` key: put down a red finish block
- `ESC` key: finish creating the track and run the simulationn

The track created will be saved in whatever file was specified under `track_file` in [config.py](./gym_car_race/config.py) or any other overriding configuration.

If you choose to run the simulation without rendering the car, you can omit `env.render()` from the main loop in your project.

## Configuring

Much of the general configuration can be done through editing the values in [config.py](./gym_car_race/config.py). A custom reward function can also be passed into the environment by defining it like the following example and adding a reference to it in the `reward.function` field in the config file:

```python
@Car.reward_function
def my_reward_func(car):
    # Function must take an instance of the car as a paramter
    ...
    # Calculate reward here using values from car

    return reward

# Make sure to add my_reward_func to either config.py or an
# overidden config

car = Car()
env.add_car(car)
```

If no reward function is provided, a default implementation is used using the corresponding values in [config.py](./gym_car_race/config.py).

For an example on how to specify and train multiple configurations at the same time, see [test_bench.py](./test_bench.py).

## Logging

Logging is handled automatically by Tensorboard. To view the logged data and visualizations, run `tensorboard dev upload --logdir './my_log_dir'` where `my_log_dir` is directory of the logs you want to view. Navigate to url from the output of the above command to see the graphs.
