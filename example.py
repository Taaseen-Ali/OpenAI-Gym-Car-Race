from SelfDriveEnv import Car, Track
from stable_baselines3.common.env_checker import check_env

block_width = 50
block_height = 50
num_blocks_x = 10
num_blocks_y = 10

env = Track(num_blocks_x, num_blocks_y, block_width, block_height)
car = Car(300, 100, 10)
env.add_car(car)
""" check_env(env) """
env.reset()

done = False
total_reward = 0

print("Running simulation...")

""" for i in range(1000): """
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total_reward += reward    
    # This is optional
    env.render()

print("Car finished with a reward of", total_reward)