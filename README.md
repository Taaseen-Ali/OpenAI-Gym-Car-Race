<img src="./logo.svg"  align="right" width="30%"/>

# OpenAI Gym Car Race

An OpenAI Gym environment for simulating a self driving car on a race track. Built for [NYU SelfDrive](https://engineering.nyu.edu/research/vertically-integrated-projects/vip-teams/nyu-self-drive).

This project is still under active development. [Bug reports](https://github.com/Taaseen-Ali/OpenAI-Gym-Car-Race/issues) are welcome.

## Prerequisites

Ensure that you are using python3 and that all required python modules are installed by running the following:

`pip install --no-cache-dir -r requirements.txt`

## Running

Run `python example.py` to execute the example project. Doing so will open a window inside which you can click on the purple blocks to create a track for the car to navigate through. After doing so, press the escape key to begin running the simulation. If you choose to run the simulation without rendering the car, you can omit `env.render()` from the main loop in your project.

## Configuring

Much of the general configuration can be done through editing the values in [config.py](./config.py). A custom reward function can also be passed into the environment by defining it like the following example and adding a reference to it using the `reward.function` field in the config file:

```python
@Car.reward_function
def my_reward_func(car):
    # Function must take an instance of the car as a paramter
    ...
    # Calculate reward here using values from car

    return reward

# Make sure to add my_reward_func to config.py

car = Car()
env.add_car(car)
```

If no reward function is provided, a default implementation is used using the corresponding values in [config.py](./config.py).
