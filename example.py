import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from gym_car_race.SelfDriveEnv import Car, Track
from gym_car_race.training_utils import TensorboardCallback, linear_schedule
from gym_car_race.config import cfg


# Set the directory where your models should be stored as well as the name of
# the model that you want to load/save

model_dir = "./models/"
model_name = "model1"
os.makedirs(model_dir, exist_ok=True)

# Set up the environment using the values found in configs

env = Track()
car = Car()
env.add_car(car)

# Uncomment this if you've made any changes to the environment and want to make
# sure that everything is still okay (no output means everything is fine):

# check_env(env)

# Uncomment one of the following depending on what you'd like to do

# A. Use an existing model
# model = PPO.load(model_dir + model_name)

# B. Create and train a new model
timesteps = 10000
model = PPO('MlpPolicy', env, tensorboard_log="./ppo/", verbose=1)
model.learn(total_timesteps=timesteps, callback=TensorboardCallback()) 
model.save(model_dir + model_name)

# Reset the env

env = Track()
car = Car()
env.add_car(car)

obs = env.reset(new=False) # You can set new=True if you'd like to create a new track

# Run the simulation until the car crashes or finishes

done = False
while not done:
    action, _states = model.predict(obs) 
    obs, rewards, done, info = env.step(action)
    env.render()
