from SelfDriveEnv import Car, Track
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

block_width = 50
block_height = 50
num_blocks_x = 10
num_blocks_y = 10

env = Track(num_blocks_x, num_blocks_y, block_width, block_height)
car = Car(300, 100, 10)
env.add_car(car)

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=40000)


env = Track(num_blocks_x, num_blocks_y, block_width, block_height)
car = Car(300, 100, 10)
env.add_car(car)
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs) 
    obs, rewards, done, info = env.step(action)
    env.render()




""" check_env(env) """
""" env.reset()

done = false
total_reward = 0

print("running simulation...")

for i in range(1000):
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total_reward += reward    
    # this is optional
    env.render()

print("car finished with a reward of", total_reward) """