import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from SelfDriveEnv import Car, Track
from training_utils import TensorboardCallback, linear_schedule
from config import cfg


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

model = None
# Function to use an existing model
def existing_model():
    global model
    model = PPO.load(model_dir + model_name)

# Function to create and train a new model
def new_model():
    global model
    timesteps = 10000
    model = PPO('MlpPolicy', env, tensorboard_log="./ppo/", verbose=1)
    model.learn(total_timesteps=timesteps, callback=TensorboardCallback()) 
    model.save(model_dir + model_name)

# # Function to load an existing model and keep training with it
# def train_existing():
#     global model
#     model = PPO.load(model_dir + model_name)
#     model.learn(total_timesteps=10000) 
#     model.save(model_dir + model_name)

# switch-case function
def mode(letter):
    switch = {
        "A": existing_model,
        "B": new_model
        # "C": train_existing
    }
    if letter not in switch:
        raise ValueError("mode() needs a letter A, B")
    switch[letter]()

# choose from the following
# type 'A' for Use an existing model
# type 'B' to create and train a new model
# type 'C' to load an existing model and keep training with it
mode('B')

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
